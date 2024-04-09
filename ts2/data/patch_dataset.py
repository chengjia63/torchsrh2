import os
import math
import random
import logging
from os.path import join as opj
from abc import ABC
from typing import Optional, List, TypedDict, Any, NamedTuple, Dict
from functools import partial

import pandas
import torch
from tqdm import tqdm

from torchsrh.datasets.common import get_chnl_min, get_chnl_max

from ts2.data.db_improc import process_read_memmap
from ts2.data.balanceable_dataset import BalanceableBaseDataset


class PatchDataset(BalanceableBaseDataset):
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
                 process_read_im: callable = process_read_memmap,
                 balance_instance_class=False,
                 **kwargs) -> None:

        super().__init__(**kwargs)
        self.data_root_ = data_root
        self.transform_ = transform
        self.target_transform_ = target_transform
        self.process_read_im_ = process_read_im
        self.num_transforms_ = num_transforms

        self.classes_ = []
        self.class_to_idx_ = {}
        self.weights_ = []

        self.instances_ = instances
        self.tensor_shape_map = tensor_shape_map

        if len(self.instances_) == 0:
            logging.warning("dataset empty")

        if balance_instance_class:
            self.replicate_balance_instances()
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
            mmap_info = self.tensor_shape_map[inst["slide_name"]]
            im = self.process_read_im_(self.make_im_path(mmap_info["path"]),
                                       tuple(mmap_info["shape"]),
                                       inst["patch_idx"])
        except:
            logging.error("bad_file - {}".format(inst.im_path))
            return {"image": None, "label": None, "path": [None]}

        im = torch.stack(
            [self.transform_(im) for _ in range(self.num_transforms_)])
        if self.target_transform_ is not None:
            target = self.target_transform_(target)

        return {"image": im, "label": target}
