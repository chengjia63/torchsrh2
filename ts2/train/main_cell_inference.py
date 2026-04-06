import logging
from typing import Optional, List, Dict, Any, Callable
from os.path import join as opj
import json

import pandas
import numpy as np

import itertools

import torch
from tqdm import tqdm
import einops
import math
from ts2.data.db_improc import MemmapTileReader
from ts2.data.balanceable_dataset import BalanceableBaseDataset
from itertools import chain
import random
import os
import pandas as pd
from torch.utils.data import DataLoader

from ts2.data.transforms import HistologyTransform, SingleCellPerturbedEvalTransform
import yaml

import pytorch_lightning as pl
from ts2.train.infra import parse_args, read_process_cf, setup_infra_testing, get_rank
from ts2.data.slide_dataset import SRHPatchCoordMapper
from ts2.lm.dinov2_eval_system import Dinov2EvalSystem


def parse_tuple_string(s: str) -> tuple[int, ...]:
    s = s.strip()
    if not (s.startswith("(") and s.endswith(")")):
        raise ValueError("Input must be a string representation of a tuple")

    content = s[1:-1].strip()
    if not content:
        return ()

    return tuple(int(item.strip()) for item in content.split(","))


class SingleCellListInferenceDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        data_root: str,
        cell_instances: str,
        transform: Callable,
        tile_size=48,
        label_key: str = "label",
        min_cells_per_slide: Optional[int] = None,
        subsample_k_per_slide: Optional[int] = None,
        subsample_seed: int = 0,
        process_read_im: Callable = MemmapTileReader(which_set="srh"),
    ):

        super().__init__()

        assert transform is not None

        self.data_root_ = data_root
        self.proposals_ = {}
        self.transform_ = transform
        self.process_read_im_ = process_read_im
        self.tile_size_ = tile_size
        self.label_key_ = label_key

        insts = self._load_instances(cell_instances)

        insts = self._filter_instances_by_min_cells_per_slide(
            insts,
            k=min_cells_per_slide,
        )

        self.instances_, self.adversarial_instances_ = self._build_instances(
            insts=insts,
            subsample_k_per_slide=subsample_k_per_slide,
            subsample_seed=subsample_seed,
        )

        logging.info(f"Num instances {len(self.instances_)}")
        logging.info(self.instances_[[self.nion_col, "mosaic"]].value_counts())

    def _load_instances(self, cell_instances: str) -> pd.DataFrame:
        insts = pd.read_csv(cell_instances, dtype=str)
        insts["proposal"] = insts["proposal"].apply(parse_tuple_string)
        insts["tensor_shape"] = insts["tensor_shape"].apply(parse_tuple_string)
        insts["mmap_idx"] = pd.to_numeric(insts["mmap_idx"]).astype(int)

        if "nion" in insts.columns:
            self.nion_col = "nion"
        else:
            self.nion_col = "patient"
            logging.warning(
                (
                    "using patient in path is not safe.",
                    "you should include a nio number column in your dataset csv.",
                )
            )

        return insts

    def _build_instances(
        self,
        insts: pd.DataFrame,
        subsample_k_per_slide: Optional[int],
        subsample_seed: int,
    ) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        return (
            self._subsample_instances_per_slide(
                insts,
                k=subsample_k_per_slide,
                seed=subsample_seed,
            ),
            None,
        )

    def _filter_instances_by_min_cells_per_slide(
        self,
        insts: pd.DataFrame,
        k: Optional[int],
    ) -> pd.DataFrame:
        if k is None:
            return insts

        group_cols = [self.nion_col, "mosaic"]
        slide_counts = (
            insts.groupby(group_cols, sort=False)
            .size()
            .rename("num_cells")
            .reset_index()
        )
        keep_slides = slide_counts[slide_counts["num_cells"] >= k][group_cols]
        filtered = insts.merge(keep_slides, on=group_cols, how="inner")

        logging.info(
            "Filtered cell instances by min_cells_per_slide=%d: %d -> %d rows across %d -> %d slides",
            k,
            len(insts),
            len(filtered),
            len(slide_counts),
            len(keep_slides),
        )
        return filtered.reset_index(drop=True)

    def _subsample_instances_per_slide(
        self,
        insts: pd.DataFrame,
        k: Optional[int],
        seed: int,
    ) -> pd.DataFrame:
        if k is None:
            return insts

        group_cols = [self.nion_col, "mosaic"]

        insts_with_order = insts.copy()
        insts_with_order["_row_order"] = np.arange(len(insts_with_order))

        sampled = (
            insts_with_order.groupby(group_cols, group_keys=False, sort=False)
            .apply(
                lambda group: (
                    group if len(group) <= k else group.sample(n=k, random_state=seed)
                )
            )
            .sort_values("_row_order")
            .drop(columns="_row_order")
            .reset_index(drop=True)
        )

        logging.info(
            "Subsampled cell instances per slide with k=%d seed=%d: %d -> %d rows",
            k,
            seed,
            len(insts),
            len(sampled),
        )
        return sampled

    def __len__(self):
        return len(self.instances_)

    def _read_image_from_inst(self, inst: pd.Series) -> torch.Tensor:
        rc_idx = (
            inst["proposal"][0] - self.tile_size_ // 2,
            inst["proposal"][1] - self.tile_size_ // 2,
        )

        mmap_path = opj(
            self.data_root_,
            inst["institution"],
            inst[self.nion_col],
            inst["mosaic"],
            "patches",
            f"{inst[self.nion_col]}-{inst['mosaic']}-patches.dat",
        )

        return self.process_read_im_(
            mmap_path, inst["tensor_shape"], [inst["mmap_idx"]], rc_idx, self.tile_size_
        )

    def _get_item_from_inst(self, inst: pd.Series) -> Dict[str, Any]:
        im = self._read_image_from_inst(inst)

        out = inst.to_dict()
        out["image"] = self.transform_(im.squeeze()).unsqueeze(0)
        out["path"] = f"{out['patch']}#{out['proposal'][0]}_{out['proposal'][1]}"
        out["label"] = inst[self.label_key_]

        return out

    def __getitem__(self, idx: int):
        """Retrieve a list of patches, from the wholeslide specified by idx"""
        return self._get_item_from_inst(self.instances_.iloc[idx])


class SingleCellListInferenceDatasetAdversarial(SingleCellListInferenceDataset):

    def __init__(
        self,
        data_root: str,
        cell_instances: str,
        transform: Callable,
        tile_size=48,
        label_key: str = "label",
        min_cells_per_slide: Optional[int] = None,
        subsample_k_per_slide: Optional[int] = None,
        subsample_seed: int = 0,
        shuffle_mode: str = "all",
        process_read_im: Callable = MemmapTileReader(which_set="srh"),
    ):

        self.shuffle_mode_ = shuffle_mode
        super().__init__(
            data_root=data_root,
            cell_instances=cell_instances,
            transform=transform,
            tile_size=tile_size,
            label_key=label_key,
            min_cells_per_slide=min_cells_per_slide,
            subsample_k_per_slide=subsample_k_per_slide,
            subsample_seed=subsample_seed,
            process_read_im=process_read_im,
        )

    def _shuffle_adversarial_instances(self, insts: pd.DataFrame) -> pd.DataFrame:
        if self.shuffle_mode_ == "all":
            out = insts.copy()
            if len(out) > 1:
                order = np.random.permutation(len(out))
                out.loc[:, :] = insts.iloc[order].to_numpy()
            print(insts)
            print(out)
            return out

        raise NotImplementedError()
        if self.shuffle_mode_ == "label":
            group_cols = [self.label_key_]
        elif self.shuffle_mode_ in {"slide_id", "mosaic"}:
            group_cols = [self.nion_col, "mosaic"]
        elif self.shuffle_mode_ in {"slide_id_patch_name", "mosaic_patch"}:
            group_cols = [self.nion_col, "mosaic", "patch"]
        else:
            raise ValueError(f"Unknown shuffle_mode: {self.shuffle_mode_}")

        out = insts.copy()
        for _, idxs in insts.groupby(group_cols, sort=False).groups.items():
            idxs = np.asarray(list(idxs))
            if len(idxs) <= 1:
                continue
            rotated = np.random.permutation(idxs)
            out.loc[idxs, :] = insts.loc[rotated].to_numpy()
        return out

    def _build_instances(
        self,
        insts: pd.DataFrame,
        subsample_k_per_slide: Optional[int],
        subsample_seed: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        insts = insts.reset_index(drop=True)
        adversarial_insts = self._shuffle_adversarial_instances(insts).reset_index(
            drop=True
        )
        sampled_insts = self._subsample_instances_per_slide(
            insts.assign(_sample_row_idx=np.arange(len(insts))),
            k=subsample_k_per_slide,
            seed=subsample_seed,
        )
        sampled_adv = adversarial_insts.iloc[
            sampled_insts["_sample_row_idx"]
        ].reset_index(drop=True)

        return (
            sampled_insts.drop(columns="_sample_row_idx").reset_index(drop=True),
            sampled_adv,
        )

    def __getitem__(self, idx: int):
        clean_inst = self.instances_.iloc[idx]
        adv_inst = self.adversarial_instances_.iloc[idx]

        clean = clean_inst.to_dict()
        pair_im = torch.stack(
            [
                self.transform_.base_aug(
                    self._read_image_from_inst(clean_inst).squeeze()
                ),
                self.transform_.base_aug(
                    self._read_image_from_inst(adv_inst).squeeze()
                ),
            ],
            dim=0,
        )
        image = self.transform_.strong_aug(pair_im)
        clean["path"] = (
            f"{clean['patch']}#{clean['proposal'][0]}_{clean['proposal'][1]}"
        )
        clean["label"] = clean_inst[self.label_key_]
        clean["bg_path"] = (
            f"{adv_inst['patch']}#{adv_inst['proposal'][0]}_{adv_inst['proposal'][1]}"
        )

        return {
            **clean,
            "image": image,
        }


from collections import defaultdict


def merge_list_of_dicts(dict_list):
    merged = defaultdict(list)
    for d in dict_list:
        for k, v in d.items():
            merged[k].extend(v)
    return dict(merged)


def main():
    cf = read_process_cf(parse_args())

    logging.info("Doing inference")

    eval_root, pred_dir, results_dir, pred_fname = setup_infra_testing(
        cf, embedded_exp_root=None
    )

    con_exp = Dinov2EvalSystem(**cf.lightning_module.params)

    transform_classes = {
        "HistologyTransform": HistologyTransform,
    }

    xform_which = cf.data.xform_params.get("which", "HistologyTransform")
    xform_params = {k: v for k, v in cf.data.xform_params.items() if k != "which"}

    dsets = {
        "SingleCellListInferenceDataset": SingleCellListInferenceDataset,
        "SingleCellListInferenceDatasetAdversarial": SingleCellListInferenceDatasetAdversarial,
    }
    dataset = dsets[cf.data.test_dataset.which](
        transform=transform_classes[xform_which](**xform_params),
        **cf.data.test_dataset.params,
    )

    data_loader = (DataLoader(dataset, **cf.data.loader.params.test),)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inf_trainer = pl.Trainer(
        accelerator=device,
        devices=1,
        default_root_dir=eval_root,
        inference_mode=True,
        deterministic=True,
    )

    pred_raw = inf_trainer.predict(con_exp, dataloaders=data_loader)

    torch.save(merge_list_of_dicts(pred_raw), opj(pred_dir, "pred.pt"))


if __name__ == "__main__":
    main()
