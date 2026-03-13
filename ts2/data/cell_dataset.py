import os
import math
import numpy as np
import time

import random
from collections import defaultdict
import logging
from os.path import join as opj
from abc import ABC
from typing import Optional, List, TypedDict, Any, NamedTuple, Dict
from functools import partial

import pandas as pd
import torch
from tqdm import tqdm

import torch.nn as nn

from torchsrh.datasets.common import get_chnl_min, get_chnl_max

from ts2.data.db_improc import MemmapReader
from ts2.data.balanceable_dataset import BalanceableBaseDataset


class CellBenchDataset(BalanceableBaseDataset):
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
                 slides_file: List,
                 transform: callable,
                 target_transform: callable = torch.tensor,
                 balance_instance_class=False,
                 num_sample_wo_replacement=None,
                 class_filter=None,
                 **kwargs) -> None:

        super().__init__(**kwargs)
        self.data_root_ = data_root
        self.transform_ = transform
        self.target_transform_ = target_transform
        self.classes_ = []
        self.class_to_idx_ = {}
        self.weights_ = []

        
        slides = pd.read_csv(slides_file)
        all_meta = []
        all_images = []
        all_paths = []


        def conditional_sample_indices(group, group_name):
            if False: 
            #if len(group) > 64:
            #if group_name in ["glioma/tumor_cells", "metastatic/adenocarcinoma", "metastatic/melanoma", "metastatic/sarcoma", "metastatic/squamous_cell"] and len(group) > 8:
                return group.sample(n=64, replace=False).index
            return group.index


        for _, s in slides.iterrows():
            meta = pd.read_csv(f"{data_root}/scbench_processed/{s['mosaic']}.csv")

            sampled_indices = meta.groupby('annot_labels').apply(lambda grp: conditional_sample_indices(grp, grp.name)).explode().tolist()

            all_meta.append(meta.iloc[sampled_indices]["annot_labels"])

            all_images.append(
                torch.load(f"{data_root}/scbench_processed/{s['mosaic']}.pt")[sampled_indices].to(torch.float))
            
            all_paths.extend([f"scbench.{s['ttype']}.{s['mosaic']}@{i}" for i in sampled_indices])


        all_meta = pd.concat(all_meta)
        all_images = torch.cat(all_images)

        class_filter = all_meta.isin(class_filter)
        all_meta = all_meta[class_filter]
        all_images[class_filter.tolist()]
        all_paths = [p for p,i in zip(all_paths, class_filter) if i]

        self.instances_ = [{"label":i,"image":j, "path":k}
                           for i,j,k in zip(all_meta, all_images, all_paths)]

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

        im = inst["image"]

        im = torch.stack([self.transform_(im)])
        
        if self.target_transform_ is not None:
            target = self.target_transform_(target)

        return {
            "image": im.contiguous(),
            "label": target,
            "path": [inst["path"]]
        }



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
                 tumor_normal_only:bool = False,
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

        if tumor_normal_only:
            for i in self.instances_:
                if not (i["label"] == "normal"):
                    i["label"] = "tumor"

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
        #im = torch.randint(high = 65536, size=(2,64,64))
        #logging.info(inst)
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

def shuffled_adversarial_instances(instances, shuffle_mode):
    """
    Return a shuffled copy of `instances` according to shuffle_mode.

    Modes
    -----
    "all"                  : fully shuffle all instances
    ""                     : no shuffle
    "label"                : shuffle within each label
    "slide_id"             : shuffle within each slide_id
    "slide_id_patch_name"  : shuffle within each (slide_id, patch_name)
    """
    if shuffle_mode == "":
        return instances.copy()

    if shuffle_mode == "all":
        out = instances.copy()
        random.shuffle(out)
        return out

    if shuffle_mode == "label":
        key_fn = lambda x: x["label"]
    elif shuffle_mode == "slide_id":
        key_fn = lambda x: x["slide_id"]
    elif shuffle_mode == "slide_id_patch_name":
        key_fn = lambda x: (x["slide_id"], x["patch_name"])
    else:
        raise ValueError(f"Unknown shuffle_mode: {shuffle_mode}")

    groups = defaultdict(list)
    for i, inst in enumerate(instances):
        groups[key_fn(inst)].append(i)
    import pdb; pdb.set_trace()
    out = instances.copy()
    for idxs in groups.values():
        vals = [instances[i] for i in idxs]
        random.shuffle(vals)
        for i, v in zip(idxs, vals):
            out[i] = v

    return out

class CellDatasetTripletEval(CellDataset):
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
                 shuffle_mode="label",
                 balance_instance_class=False,
                 num_sample_wo_replacement=None,
                 **kwargs) -> None:

        super().__init__(balance_instance_class=balance_instance_class, num_sample_wo_replacement=num_sample_wo_replacement,**kwargs)

        assert not balance_instance_class
        assert not num_sample_wo_replacement
        assert self.num_transforms_==1
        logging.info(self.transform_.strong_aug)
        assert type(self.transform_.strong_aug) is nn.Identity


        self.adversarial_instances_ = shuffled_adversarial_instances(
            self.instances_, shuffle_mode
        )


    def __len__(self):
        return len(self.instances_)

    def make_im_path(self, x):
        return opj(self.data_root_, x)

    def get_item_impl(self, inst):
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

        return {
            "image": im,
            "label": target,
            "path":
            [f"{inst['slide_id']}-{inst['patch_name']}@{inst['cell_idx']}"]
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        inst = self.instances_[idx]
        target = self.class_to_idx_[inst["label"]]

        clean_im = self.get_item_impl(inst)
        adv_im = self.get_item_impl(self.adversarial_instances_[idx])

        im = torch.concat([clean_im["image"], adv_im["image"]])

        if self.target_transform_ is not None:
            target = self.target_transform_(target)

        return {
            "image": im,
            "label": target,
            "path": [f"{clean_im['path']}/{adv_im['path']}"]
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
        #return (self.transform_(torch.randint(high = 65536, size=(2,64,64)).to(float)), 0)

        #start = time.time()
        inst = self.instances_[idx]
        #logging.info({
        #    'slide_id': inst["slide_id"],
        #    'patch_name': inst["patch_name"],
        #    'cell_idx': inst["cell_idx"], 
        #})
        target = self.class_to_idx_[inst["label"]]

        try:
            mmap_info = self.tensor_shape_map[inst["slide_id"]]
            mmap_path = self.make_im_path(mmap_info["path"])
            im = self.process_read_im_(mmap_path, tuple(mmap_info["shape"]),
                                       inst["cell_idx"])
        except:
            logging.error("bad_file - {}".format(inst.im_path))
            return {"image": None, "label": None, "path": [None]}


        #end1 = time.time()

        im = self.transform_(im)
        if self.target_transform_ is not None:
            target = self.target_transform_(target)

        #end2 = time.time()

        #logging.info({
        #    'slide_id': inst["slide_id"],
        #    'patch_name': inst["patch_name"],
        #    'cell_idx': inst["cell_idx"], 
        #    "time0": end1 - start,
        #    "time1": end2 - start,
        #    "times": times
        #})
        return im, target
