import os
import math
import numpy as np

import random
import logging
from os.path import join as opj
from abc import ABC
from typing import Optional, List, TypedDict, Any, NamedTuple, Dict
from functools import partial

import pandas as pd
import torch
from tqdm import tqdm

from torchsrh.datasets.common import get_chnl_min, get_chnl_max

from ts2.data.db_improc import MemmapReader
from ts2.data.balanceable_dataset import BalanceableBaseDataset


class CellDataset(BalanceableBaseDataset):
    """Patch Base Dataset.

    Patch datasets treats each patch to be independent

    Attributes:
        data_root_: str containing the root path of the dataset
        transform_: transformations to be performed on the data
        target_transform_: transformations to be performed on the labels
        df_: data frame containing patch information
        instances_: a list of Slide
        classes_: a set of primary labels in the dataset (could be any
            hashable object)
        class_to_idx_: a mapping from the primary class label to a numeric
            label [0 .. num classes - 1]
        weights_: weights assigned to each class, inverse proportional to the
            slide count in each class
    """

    def __init__(self,
                 data_root: str,
                 instances: List,
                 tensor_shape_map: Dict,
                 transform: callable,
                 target_transform: callable = torch.tensor,
                 num_transforms: int = 1,
                 process_read_im: callable = MemmapReader("srh"),
                 balance_instance_class=False,
                 num_sample_wo_replacement=None,
                 **kwargs) -> None:

        super().__init__(**kwargs)
        self.data_root_ = data_root
        self.transform_ = transform
        self.target_transform_ = target_transform
        self.process_read_im_ = process_read_im
        self.num_transforms_ = num_transforms
        self.num_samples_ = 1  # a hack for hidisc support
        self.classes_ = []
        self.class_to_idx_ = {}
        self.weights_ = []

        self.instances_ = instances
        self.tensor_shape_map = tensor_shape_map

        if len(self.instances_) == 0:
            logging.warning("dataset empty")

        if balance_instance_class:
            self.replicate_balance_instances()

        if num_sample_wo_replacement:
            self.sample_instances_wo_replacement(num_sample_wo_replacement)

        self.get_weights()

        logging.info(self.transform_)

    def __len__(self):
        return len(self.instances_)

    def make_im_path(self, x):
        return opj(self.data_root_, x)

    def __getitem__(self, idx: int) -> Dict[str, Any]:

        inst = self.instances_[idx]
        target = self.class_to_idx_[inst["label"]]

        try:
            mmap_info = self.tensor_shape_map[inst["slide_id"]]
            mmap_path = self.make_im_path(mmap_info["path"])
            im = self.process_read_im_(mmap_path, tuple(mmap_info["shape"]),
                                       inst["cell_idx"])
        except:
            logging.error("bad_file - {}".format(inst.im_path))
            return {"image": None, "label": None, "path": [None]}

        im = torch.stack(
            [self.transform_(im) for _ in range(self.num_transforms_)])
        if self.target_transform_ is not None:
            target = self.target_transform_(target)

        return {
            "image": im,
            "label": target,
            "path":
            [f"{inst['slide_id']}-{inst['patch_name']}@{inst['cell_idx']}"]
        }


class CellPatchDataset(BalanceableBaseDataset):
    """Patch Base Dataset.

    Patch datasets treats each patch to be independent

    Attributes:
        data_root_: str containing the root path of the dataset
        transform_: transformations to be performed on the data
        target_transform_: transformations to be performed on the labels
        df_: data frame containing patch information
        instances_: a list of Slide
        classes_: a set of primary labels in the dataset (could be any
            hashable object)
        class_to_idx_: a mapping from the primary class label to a numeric
            label [0 .. num classes - 1]
        weights_: weights assigned to each class, inverse proportional to the
            slide count in each class
    """

    def __init__(self,
                 data_root: str,
                 instances: List,
                 tensor_shape_map: Dict,
                 transform: callable,
                 target_transform: callable = torch.tensor,
                 num_transforms: int = 1,
                 process_read_im: callable = MemmapReader("srh"),
                 **kwargs) -> None:

        super().__init__(**kwargs)
        self.data_root_ = data_root
        self.transform_ = transform
        self.target_transform_ = target_transform
        self.process_read_im_ = process_read_im
        self.num_transforms_ = num_transforms
        self.tensor_shape_map = tensor_shape_map

        #self.num_samples_ = 1  # a hack for hidisc support
        #self.classes_ = []
        #self.class_to_idx_ = {}
        #self.weights_ = []

        self.instances_ = instances

        if len(self.instances_) == 0:
            logging.warning("dataset empty")

        instance_df = pd.DataFrame(self.instances_)
        #instance_df = instance_df.groupby(["slide_id", "patch_name", "label"]).agg(list).sample(100).explode(["cell_idx"]).reset_index()[["slide_id", "patch_name", "cell_idx", "label"]]

        self.instance_patch_df = instance_df.groupby(
            ["slide_id", "patch_name", "label"]).agg(list).reset_index()

        #self.classes = instance_df["label"].unique()
        #self.tumor_classes = sorted(set(self.classes.tolist()).difference({"normal"}))
        #
        #self.cells_by_class = {c:
        #    instance_df[instance_df["label"]==c]
        #    for c in self.classes
        #    }

        logging.info(self.transform_)

    def __len__(self):
        return len(self.instance_patch_df)

    def make_im_path(self, x):
        return opj(self.data_root_, x)

    def get_im(self, inst):
        mmap_info = self.tensor_shape_map[inst["slide_id"]]
        mmap_path = self.make_im_path(mmap_info["path"])
        im = self.process_read_im_(mmap_path, tuple(mmap_info["shape"]),
                                   inst["cell_idx"])
        return im

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        curr_patch = self.instance_patch_df.iloc[[idx]].explode("cell_idx")

        curr_pixels = [self.get_im(i) for _, i in curr_patch.iterrows()]
        curr_pixels = torch.stack([self.transform_(i) for i in curr_pixels])

        target = 1 - (self.instance_patch_df.iloc[idx]["label"] == "normal")
        if self.target_transform_ is not None:
            target = self.target_transform_(target)

        inst = curr_patch.iloc[0]

        return {
            "path": f"{inst['slide_id']}-{inst['patch_name']}",
            "pixels": curr_pixels,
            "label": target
        }


class CellBagDataset(BalanceableBaseDataset):
    """Patch Base Dataset.

    Patch datasets treats each patch to be independent

    Attributes:
        data_root_: str containing the root path of the dataset
        transform_: transformations to be performed on the data
        target_transform_: transformations to be performed on the labels
        df_: data frame containing patch information
        instances_: a list of Slide
        classes_: a set of primary labels in the dataset (could be any
            hashable object)
        class_to_idx_: a mapping from the primary class label to a numeric
            label [0 .. num classes - 1]
        weights_: weights assigned to each class, inverse proportional to the
            slide count in each class
    """

    def __init__(self,
                 data_root: str,
                 instances: List,
                 tensor_shape_map: Dict,
                 normal_p,
                 ncell_mean,
                 ncell_std,
                 ncell_min,
                 mock_length,
                 transform: callable,
                 target_transform: callable = torch.tensor,
                 num_transforms: int = 1,
                 process_read_im: callable = MemmapReader("srh"),
                 **kwargs) -> None:

        super().__init__(**kwargs)
        self.data_root_ = data_root
        self.transform_ = transform
        self.target_transform_ = target_transform
        self.process_read_im_ = process_read_im
        self.num_transforms_ = num_transforms
        #self.num_samples_ = 1  # a hack for hidisc support
        #self.classes_ = []
        #self.class_to_idx_ = {}
        #self.weights_ = []

        self.instances_ = instances

        if len(self.instances_) == 0:
            logging.warning("dataset empty")

        self.tensor_shape_map = tensor_shape_map
        self.normal_p = normal_p
        self.ncell_mean = ncell_mean
        self.ncell_std = ncell_std
        self.ncell_min = ncell_min

        self.mock_length = mock_length
        instance_df = pd.DataFrame(self.instances_)

        self.classes = np.sort(instance_df["label"].unique())
        self.tumor_classes = sorted(
            set(self.classes.tolist()).difference({"normal"}))

        self.cells_by_class = {
            c: instance_df[instance_df["label"] == c]
            for c in self.classes
        }

        logging.info(self.transform_)

    def __len__(self):
        return self.mock_length

    def make_im_path(self, x):
        return opj(self.data_root_, x)

    def get_im(self, inst):
        mmap_info = self.tensor_shape_map[inst["slide_id"]]
        mmap_path = self.make_im_path(mmap_info["path"])
        im = self.process_read_im_(mmap_path, tuple(mmap_info["shape"]),
                                   inst["cell_idx"])
        return im

    def __getitem__(self, idx: int) -> Dict[str, Any]:

        curr_ncells = max(
            self.ncell_min,
            int(np.random.randn() * self.ncell_std + self.ncell_mean))

        if random.random() < self.normal_p:
            curr_class = "normal"
            curr_nnormal = curr_ncells
            curr_ntumor = 0
            target = 0

        else:
            curr_class = random.sample(self.tumor_classes, k=1)[0]
            curr_ntumor = max(int(random.random() * curr_ncells), 1)
            curr_nnormal = curr_ncells - curr_ntumor
            target = 1

        curr_tumor_cells = self.cells_by_class[curr_class].sample(curr_ntumor)
        curr_normal_cells = self.cells_by_class["normal"].sample(curr_nnormal)

        if len(curr_tumor_cells):
            curr_tumor_pixels = [
                self.get_im(i) for _, i in curr_tumor_cells.iterrows()
            ]
        else:
            curr_tumor_pixels = []

        if len(curr_normal_cells):
            curr_normal_pixels = [
                self.get_im(i) for _, i in curr_normal_cells.iterrows()
            ]
        else:
            curr_normal_pixels = []

        curr_pixels = torch.stack(
            [self.transform_(i) for i in curr_tumor_pixels] +
            [self.transform_(i) for i in curr_normal_pixels])

        if self.target_transform_ is not None:
            target = self.target_transform_(target)

        return {"path": "bag", "pixels": curr_pixels, "label": target}


class CellDatasetDINOv2(CellDataset):
    """Patch Base Dataset.

    Patch datasets treats each patch to be independent

    Attributes:
        data_root_: str containing the root path of the dataset
        transform_: transformations to be performed on the data
        target_transform_: transformations to be performed on the labels
        df_: data frame containing patch information
        instances_: a list of Slide
        classes_: a set of primary labels in the dataset (could be any
            hashable object)
        class_to_idx_: a mapping from the primary class label to a numeric
            label [0 .. num classes - 1]
        weights_: weights assigned to each class, inverse proportional to the
            slide count in each class
    """

    def __init__(self, **kwargs) -> None:

        super().__init__(**kwargs)
        assert self.num_transforms_ == 0

    def __getitem__(self, idx: int) -> Dict[str, Any]:

        inst = self.instances_[idx]
        target = self.class_to_idx_[inst["label"]]

        try:
            mmap_info = self.tensor_shape_map[inst["slide_id"]]
            mmap_path = self.make_im_path(mmap_info["path"])
            im = self.process_read_im_(mmap_path, tuple(mmap_info["shape"]),
                                       inst["cell_idx"])
        except:
            logging.error("bad_file - {}".format(inst.im_path))
            return {"image": None, "label": None, "path": [None]}

        im = self.transform_(im)
        if self.target_transform_ is not None:
            target = self.target_transform_(target)

        return im, target
