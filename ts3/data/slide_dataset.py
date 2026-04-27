import os
from os.path import join as opj
import math
import random
import logging
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

tqdm.pandas()


class LabelIndexTensorTransform:
    def __init__(self, class_to_idx: dict):
        self.class_to_idx_ = class_to_idx

    def __call__(self, label):
        if label not in self.class_to_idx_:
            raise KeyError(f"Unknown label {label!r}")
        return torch.tensor(self.class_to_idx_[label], dtype=torch.long)


class BalanceableBaseDataset(Dataset):
    def __init__(
        self,
        label_key: str = "label",
        eps: float = 1.0e-6,
        classes=None,
        class_order=None,
        class_to_idx=None,
    ):
        super().__init__()
        self.label_key_ = label_key
        self.classes_ = class_order if class_order is not None else classes
        self.class_to_idx_ = class_to_idx
        self.weights_ = None
        self.eps_ = eps

    def get_instance_label(self, inst):
        if self.label_key_ not in inst:
            raise KeyError(
                f"Instance is missing required label key {self.label_key_!r}"
            )
        return inst[self.label_key_]

    def process_classes(self):
        if not self.classes_:
            all_labels = [self.get_instance_label(i) for i in self.instances_]
            classes = set(all_labels)

            if len(set(map(type, classes))) == 1:
                self.classes_ = sorted(classes)
            else:
                classes = list(classes)
                sort_class_idx = np.argsort([str(c) for c in classes]).tolist()
                self.classes_ = [classes[i] for i in sort_class_idx]

        self.class_to_idx_ = {c: i for i, c in enumerate(self.classes_)}
        logging.info("Labels: %s", self.classes_)

    def get_count(self):
        if not self.classes_ or self.class_to_idx_ is None:
            self.process_classes()

        all_labels = [
            self.class_to_idx_[self.get_instance_label(i)] for i in self.instances_
        ]

        count = Counter(all_labels)
        count = torch.Tensor([count[i] for i in range(len(self.classes_))])
        logging.info("Count: %s", count)
        return count

    def get_weights(self):
        count = self.get_count()
        inv_count = 1.0 / (count + self.eps_)
        inv_count[count == 0] = 0
        self.weights_ = inv_count / torch.sum(inv_count)
        logging.debug("Weights: %s", self.weights_)
        return self.weights_

    def replicate_balance_instances(self):
        logging.info("replicating instances to balance each class")
        self.process_classes()
        count = self.get_count()
        val_sample = int(max(count))
        random.shuffle(self.instances_)
        all_repl_instances = []

        for label in self.classes_:
            inst_l = [i for i in self.instances_ if self.get_instance_label(i) == label]
            n_rep = math.ceil(val_sample / len(inst_l))
            all_repl_instances.extend((inst_l * n_rep)[:val_sample])

        self.instances_ = all_repl_instances


class SlideEmbeddingDataset(BalanceableBaseDataset):
    """Slide-level dataset backed by one embedding .pt file per slide.

    The CSV is the source of truth. Patient IDs and slide names are treated as
    opaque strings; this dataset does not parse NIO numbers or use TS metadata.
    """

    def __init__(
        self,
        embedding_root: str,
        slides_csv: str,
        embedding_path_template: str,
        label_col: str = "label",
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = torch.tensor,
        num_instance_self_replicate: int = 1,
        balance_instance_class: bool = False,
        **kwargs,
    ):
        super().__init__(label_key=label_col, **kwargs)
        if embedding_root is None:
            raise ValueError("embedding_root must not be None")
        self.transform_ = transform
        self.target_transform_ = target_transform
        self.label_col_ = label_col
        self.embedding_root_ = embedding_root
        if num_instance_self_replicate != 1:
            raise ValueError(
                "num_instance_self_replicate is no longer supported. "
                "Use data.splits.<split>.dataloader.sampler.repeat_factor or "
                "samples_per_epoch instead."
            )

        self.instances_ = self.load_slides_csv(
            slides_csv=slides_csv,
            label_col=label_col,
            embedding_path_template=embedding_path_template,
            embedding_root=embedding_root,
        )

        if balance_instance_class:
            self.replicate_balance_instances()
        self.get_weights()
        if (
            self.target_transform_ is torch.tensor
            and self.instances_
            and isinstance(self.instances_[0][self.label_col_], str)
        ):
            self.target_transform_ = LabelIndexTensorTransform(self.class_to_idx_)

    @staticmethod
    def load_slides_csv(
        slides_csv: str,
        label_col: str,
        embedding_path_template: str,
        embedding_root: str,
    ):
        df = pd.read_csv(slides_csv, dtype=str)
        required_cols = {"institution", "patient", "mosaic", label_col}
        missing_cols = sorted(required_cols.difference(df.columns))
        if missing_cols:
            raise ValueError(
                f"Slide CSV {slides_csv} is missing columns: {missing_cols}"
            )
        if df.empty:
            raise ValueError(f"Slide CSV {slides_csv} is empty")

        df = df.loc[:, ["institution", "patient", "mosaic", label_col]].copy()
        rows = df.to_dict(orient="records")
        df["path"] = [
            opj(embedding_root, embedding_path_template.format(**row)) for row in rows
        ]

        path_exists = df["path"].progress_apply(os.path.exists)

        if (~path_exists).any():
            dropped_df = df.loc[~path_exists]
            logging.warning(
                "Dropping %d slide rows with missing embedding files from %s:\n%s",
                len(dropped_df),
                slides_csv,
                dropped_df.to_string(index=False),
            )
            df = df.loc[path_exists].copy()

        if df.empty:
            raise ValueError(
                f"All slide rows in {slides_csv} were dropped because their embedding files do not exist under {embedding_root}"
            )

        return df.to_dict(orient="records")

    @staticmethod
    def parse_global_cell_coords(cell_path: str) -> tuple[int, int]:
        patch_name, cell_coord = cell_path.split("#", 1)
        patch_coord = patch_name.rsplit("-", 1)[1]
        patch_top_str, patch_left_str = patch_coord.split("_", 1)
        cell_r_str, cell_c_str = cell_coord.split("_", 1)
        return (
            int(round(float(patch_top_str))) + int(round(float(cell_r_str))),
            int(round(float(patch_left_str))) + int(round(float(cell_c_str))),
        )

    def __len__(self):
        return len(self.instances_)

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self.instances_):
            raise IndexError(
                f"SlideEmbeddingDataset index {idx} is out of range for "
                f"{len(self.instances_)} slides"
            )
        sample = dict(self.instances_[idx])
        data = torch.load(sample["path"], map_location="cpu")
        assert ("path" in data) and ("embeddings" in data)

        sample["embeddings"] = data["embeddings"]
        sample["coords"] = torch.tensor(
            [self.parse_global_cell_coords(path) for path in data["path"]]
        )

        if self.transform_ is not None:
            sample["embeddings"] = self.transform_(sample["embeddings"])
        if self.target_transform_ is not None:
            sample[self.label_col_] = self.target_transform_(sample[self.label_col_])

        return sample
