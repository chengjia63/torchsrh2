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

from ts2.data.transforms import HistologyTransform
import yaml

import pytorch_lightning as pl
from ts2.train.infra import parse_args, read_process_cf, setup_infra_testing, get_rank
from ts2.data.slide_dataset import SRHPatchCoordMapper
from ts2.lm.dinov2_eval_system import Dinov2EvalSystem


class SingleCellTempInferenceDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        data_root: str,
        sc_proposal_root: str,
        slide_instances: str,
        transform: Callable,
        tile_size=48,
        process_read_im: Callable = MemmapTileReader(which_set="srh"),
    ):

        super().__init__()
        raise RuntimeError("SingleCellTempInferenceDataset is deprecated.")

        default_seg_model = "03207B00"
        self.data_root_ = data_root
        self.proposals_ = {}
        self.transform_ = transform
        self.process_read_im_ = process_read_im
        self.tile_size_ = tile_size

        assert transform is not None

        slide_instances_df = pd.read_csv(slide_instances, dtype=str)
        instances = []
        for i, r in tqdm(slide_instances_df.iterrows()):

            # Getting slide / patch metadata
            pt_meta_fname = opj(
                data_root, r["institution"], r["patient"], f"{r['patient']}_meta.json"
            )

            if not os.path.exists(pt_meta_fname):
                logging.warning(f"No pt meta file - {pt_meta_fname}")
                continue

            with open(pt_meta_fname) as fd:
                slide_meta_all_inf = json.load(fd)["slides"][r["mosaic"]]["predictions"]

            if default_seg_model in slide_meta_all_inf:
                slide_meta = slide_meta_all_inf[default_seg_model]
            else:
                logging.warning(f"No {default_seg_model} inference - {pt_meta_fname}")
                continue

            all_patches = [
                (k, v, tt)
                for tt, subdict in slide_meta["patches"].items()
                for k, v in subdict.items()
            ]

            all_patches = pd.DataFrame(
                all_patches, columns=["patch_flat", "mmap_idx", "patch_type"]
            )
            all_patches.loc[:, "tensor_shape"] = [
                tuple(slide_meta["tensor_shape"])
            ] * len(all_patches)

            # Get sincle cell detection results
            s = f"{r['patient']}-{r['mosaic']}"

            slide_meta = opj(sc_proposal_root, f"{s}-meta.csv")
            assert os.path.exists(opj(sc_proposal_root, f"{s}-meta.csv"))

            try:
                cp = pd.read_csv(slide_meta)
            except pd.errors.EmptyDataError:
                logging.warning(f"No cells before filtering-- {s}")
                continue

            # casting cell coord datatypes
            cp["centroid_r"] = cp["centroid_r"].round().astype(int)
            cp["centroid_c"] = cp["centroid_c"].round().astype(int)

            # filter out undesiable cells
            cp_filt = cp[
                cp["celltype"].isin({"nuclei", "mp"})
                & (cp["score"] > 0.5)
                & (self.tile_size_ / 2 <= cp["centroid_r"])
                & (cp["centroid_r"] <= 300 - self.tile_size_ / 2)
                & (self.tile_size_ / 2 <= cp["centroid_c"])
                & (cp["centroid_c"] <= 300 - self.tile_size_ / 2)
            ]

            if len(cp_filt) == 0:
                logging.warning(f"no cells - {s}")
                continue

            # putting instances together
            cp_filt = pd.DataFrame(
                {
                    "patch": cp_filt["patch"],
                    "proposal": zip(cp_filt["centroid_r"], cp_filt["centroid_c"]),
                }
            )
            cp_filt.loc[:, "patch_flat"] = cp_filt["patch"].apply(
                SRHPatchCoordMapper.to_universal_patch_name
            )
            cp_filt.loc[:, "institution"] = r["institution"]
            cp_filt.loc[:, "patient"] = r["patient"]
            cp_filt.loc[:, "mosaic"] = r["mosaic"]

            cp_filt = pd.merge(cp_filt, all_patches, on="patch_flat", how="left")

            # if len(cp_filt) > 800:
            #    cp_filt = cp_filt.sample(800, random_state=1000)
            instances.append(cp_filt)

        self.instances_ = pd.concat(instances).reset_index(drop=True)

        logging.info(f"Num instances {len(self.instances_)}")
        logging.info(self.instances_[["patient", "mosaic"]].value_counts())

    def __len__(self):
        return len(self.instances_)

    def __getitem__(self, idx: int):
        """Retrieve a list of patches, from the wholeslide specified by idx"""
        inst = self.instances_.iloc[idx]
        rc_idx = (
            inst["proposal"][0] - self.tile_size_ // 2,
            inst["proposal"][1] - self.tile_size_ // 2,
        )

        mmap_path = opj(
            self.data_root_,
            inst["institution"],
            inst["patient"],
            inst["mosaic"],
            "patches",
            f"{inst['patient']}-{inst['mosaic']}-patches.dat",
        )

        im = self.process_read_im_(
            mmap_path, inst["tensor_shape"], [inst["mmap_idx"]], rc_idx, self.tile_size_
        )

        out = inst.to_dict()
        out["image"] = self.transform_(im.squeeze()).unsqueeze(0)

        out["path"] = f"{out['patch']}#{out['proposal'][0]}_{out['proposal'][1]}"
        out["label"] = inst["patch_type"]
        return out


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
        process_read_im: Callable = MemmapTileReader(which_set="srh"),
    ):

        super().__init__()

        assert transform is not None

        self.data_root_ = data_root
        self.proposals_ = {}
        self.transform_ = transform
        self.process_read_im_ = process_read_im
        self.tile_size_ = tile_size

        insts = pd.read_csv(cell_instances, dtype=str)
        insts["proposal"] = insts["proposal"].apply(parse_tuple_string)
        insts["tensor_shape"] = insts["tensor_shape"].apply(parse_tuple_string)
        insts["mmap_idx"] = pd.to_numeric(insts["mmap_idx"]).astype(int)
        self.instances_ = insts

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

        logging.info(f"Num instances {len(self.instances_)}")
        logging.info(self.instances_[["patient", "mosaic"]].value_counts())

    def __len__(self):
        return len(self.instances_)

    def __getitem__(self, idx: int):
        """Retrieve a list of patches, from the wholeslide specified by idx"""
        inst = self.instances_.iloc[idx]
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

        im = self.process_read_im_(
            mmap_path, inst["tensor_shape"], [inst["mmap_idx"]], rc_idx, self.tile_size_
        )

        out = inst.to_dict()
        out["image"] = self.transform_(im.squeeze()).unsqueeze(0)

        out["path"] = f"{out['patch']}#{out['proposal'][0]}_{out['proposal'][1]}"
        out["label"] = inst["patch_type"]
        return out


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

    dsets = {
        "SingleCellListInferenceDataset": SingleCellListInferenceDataset,
        "SingleCellTempInferenceDataset": SingleCellTempInferenceDataset,
    }
    dataset = dsets[cf.data.test_dataset.which](
        transform=HistologyTransform(**cf.data.xform_params),
        **cf.data.test_dataset.params,
    )

    data_loader = (DataLoader(dataset, **cf.data.loader.params.test),)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inf_trainer = pl.Trainer(
        accelerator=device, devices=1, default_root_dir=eval_root, inference_mode=True
    )  # deterministic=True)

    pred_raw = inf_trainer.predict(con_exp, dataloaders=data_loader)

    torch.save(merge_list_of_dicts(pred_raw), opj(pred_dir, "pred.pt"))


if __name__ == "__main__":
    main()
