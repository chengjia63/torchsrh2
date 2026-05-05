import functools
import os
from os.path import join as opj
import logging
from typing import Optional, Callable

import pandas as pd
import torch
from tqdm import tqdm

from ts3.data.base_dataset import BalanceableBaseDataset

tqdm.pandas()


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
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = torch.tensor,
        balance_instance_class: bool = False,
        **kwargs,
    ):
        super().__init__(label_key=label_col, **kwargs)
        if embedding_root is None:
            raise ValueError("embedding_root must not be None")
        self.label_col_ = label_col
        self.embedding_root_ = embedding_root
        self.instances_ = self.load_slides_csv(
            slides_csv=slides_csv,
            label_col=label_col,
            embedding_path_template=embedding_path_template,
            embedding_root=embedding_root,
        )

        if balance_instance_class:
            self.replicate_balance_instances()
        self.get_weights()

        self.transform_ = transform
        if isinstance(target_transform, functools.partial):
            self.target_transform_ = target_transform(class_order=self.classes_)
        else:
            self.target_transform_ = target_transform

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
            sample = self.transform_(sample)
        if self.target_transform_ is not None:
            sample[self.label_col_] = self.target_transform_(sample[self.label_col_])

        return sample
