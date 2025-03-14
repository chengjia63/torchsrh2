import logging
from multiprocessing.sharedctypes import Value
from typing import Optional, List, Dict, Any
from abc import ABC
from os.path import join as opj
import pandas
import numpy as np
import torch
from tqdm import tqdm
import einops
import math
from ts2.data.db_improc import MemmapReader
from ts2.data.balanceable_dataset import BalanceableBaseDataset
from itertools import chain
import random


class HierarchicalBaseDataset(BalanceableBaseDataset, ABC):
    """Patient Base Dataset. Abstract class.

    Attributes:
        data_root_: str containing the root path of the dataset
        image_reader_: callable that reads in images (and some processing)
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
                 target_transform: Optional[callable] = torch.tensor,
                 num_transforms: int = 1,
                 process_read_im: callable = MemmapReader("srh"),
                 num_instance_self_replicate: int = 1,
                 balance_instance_class=False,
                 num_sample_wo_replacement=None,
                 **kwargs) -> None:
        """Inits the base abstract dataset

        Populate each attribute and walk through each patient to look for patches
        """
        super().__init__(**kwargs)

        self.data_root_ = data_root
        self.transform_ = transform
        self.target_transform_ = target_transform
        self.num_transforms_ = num_transforms

        self.process_read_im_ = process_read_im

        self.classes_ = []
        self.class_to_idx_ = {}
        self.weights_ = []

        self.num_replicates_ = num_instance_self_replicate
        logging.info(f"number of replicates {self.num_replicates_}")

        self.instances_ = instances
        self.tensor_shape_map = tensor_shape_map

        assert len(self.instances_) > 0
        self.init_weights_ = self.get_weights()
        if balance_instance_class:
            self.replicate_balance_instances()
        if num_sample_wo_replacement:
            self.sample_instances_wo_replacement(num_sample_wo_replacement)
        self.get_weights()

        logging.info(self.transform_)

    def make_im_path(self, x):
        return opj(self.data_root_, x)

    def __len__(self):
        assert self.num_replicates_ > 0
        return len(self.instances_) * self.num_replicates_


class SingleLevelHierarchicalDataset(HierarchicalBaseDataset):

    def __init__(self, num_samples: int = 1, **kwargs):

        super().__init__(**kwargs)
        #self.instances_ = tuple([(i["name"], i["label"],
        #                          tuple([(p["patch_name"], p["patch_idx"])
        #                                 for p in i["patches"]]))
        #                         for i in self.instances_])

        self.num_samples_ = num_samples

    @torch.no_grad()
    def read_images(self, inst: Dict):
        """Read in a list of patches, different patches and transformations"""

        #patches_list = inst["patches"]
        im_id = random.sample(range(len(inst["patches"])), self.num_samples_)
        #mmap_id = [patches_list[i % 1000]["patch_idx"] for i in im_id]

        images = self.process_read_im_(
            self.make_im_path(self.tensor_shape_map[inst["name"]]["path"]),
            tuple(self.tensor_shape_map[inst["name"]]["shape"]), im_id)

        im_id = None

        assert self.transform_ is not None
        images = torch.stack([
            self.transform_(im) for _ in range(self.num_transforms_)
            for im in images
        ])

        return images

    def __getitem__(self, idx: int):
        """Retrieve a list of patches, from the wholeslide specified by idx"""
        idx = idx % len(self.instances_)
        instance = self.instances_[idx]
        target = self.class_to_idx_[instance["label"]]
        im = self.read_images(instance)

        if self.target_transform_ is not None:
            target = self.target_transform_(target)

        return {"image": im, "label": target}


class SingleLevelHierarchicalDatasetSingleViewDINOV2(
        SingleLevelHierarchicalDataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.num_samples_ == 1  # This version only work with 1 xform
        assert self.num_transforms_ == 0

    @torch.no_grad()
    def read_images(self, inst: Dict):
        """Read in a list of patches, different patches and transformations"""

        #patches_list = inst["patches"]
        im_id = random.sample(range(len(inst["patches"])), self.num_samples_)
        #mmap_id = [patches_list[i % 1000]["patch_idx"] for i in im_id]

        images = self.process_read_im_(
            self.make_im_path(self.tensor_shape_map[inst["name"]]["path"]),
            tuple(self.tensor_shape_map[inst["name"]]["shape"]), im_id)

        im_id = None
        assert self.transform_ is not None
        images = self.transform_(images.squeeze())
        return images

    def __getitem__(self, idx: int):
        """Retrieve a list of patches, from the wholeslide specified by idx"""
        idx = idx % len(self.instances_)
        instance = self.instances_[idx]
        target = self.class_to_idx_[instance["label"]]
        im = self.read_images(instance)

        if self.target_transform_ is not None:
            target = self.target_transform_(target)

        return im, target


SingleLevelHierarchicalDatasetDINOV2 = SingleLevelHierarchicalDatasetSingleViewDINOV2


class SingleLevelHierarchicalDatasetMultipleViewDINOV2(
        SingleLevelHierarchicalDataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.num_transforms_ == 0

    @torch.no_grad()
    def read_images(self, inst: Dict):
        """Read in a list of patches, different patches and transformations"""

        #patches_list = inst["patches"]
        im_id = random.sample(range(len(inst["patches"])), self.num_samples_)
        #mmap_id = [patches_list[i % 1000]["patch_idx"] for i in im_id]

        images = self.process_read_im_(
            self.make_im_path(self.tensor_shape_map[inst["name"]]["path"]),
            tuple(self.tensor_shape_map[inst["name"]]["shape"]), im_id)

        im_id = None
        assert self.transform_ is not None
        images = [self.transform_(i) for i in images]
        images = {
            k: list(chain(*[i[k] for i in images]))
            for k in images[0].keys()
        }
        return images

    def __getitem__(self, idx: int):
        """Retrieve a list of patches, from the wholeslide specified by idx"""
        idx = idx % len(self.instances_)
        instance = self.instances_[idx]
        target = self.class_to_idx_[instance["label"]]
        im = self.read_images(instance)

        if self.target_transform_ is not None:
            target = self.target_transform_(target)

        return im, target


class SLHDatasetWithFMEmbeddings(SingleLevelHierarchicalDataset):

    def __init__(self, fm_root: str, fm_tags: List[str], **kwargs):

        super().__init__(**kwargs)
        self.fm_root_ = fm_root
        self.fm_tags_ = fm_tags

    def make_fm_path(self, x, tag):
        return opj(
            self.fm_root_,
            *x.split("/")[1:3],
            x.split("/")[4].removesuffix("-patches.dat") + f"-embs-{tag}.dat")

    @torch.no_grad()
    def read_images(self, inst: Dict):
        """Read in a list of patches, different patches and transformations"""

        im_id = random.sample(range(len(inst["patches"])), self.num_samples_)

        rel_inst_path = self.tensor_shape_map[inst["name"]]["path"]
        fm_paths = [self.make_fm_path(rel_inst_path, t) for t in self.fm_tags_]

        tensor_shape = tuple(self.tensor_shape_map[inst["name"]]["shape"])
        ims, embs = self.process_read_im_(
            [self.make_im_path(rel_inst_path), fm_paths],
            [tensor_shape, (tensor_shape[0], -1)], im_id)

        im_id = None
        assert self.transform_ is not None

        images = torch.stack([
            self.transform_(im) for _ in range(self.num_transforms_)
            for im in ims
        ])

        embs = [em.repeat((self.num_transforms_, 1)) for em in embs]

        return images, embs

    def __getitem__(self, idx: int):
        """Retrieve a list of patches, from the wholeslide specified by idx"""
        idx = idx % len(self.instances_)
        instance = self.instances_[idx]
        target = self.class_to_idx_[instance["label"]]
        im, emb = self.read_images(instance)

        if self.target_transform_ is not None:
            target = self.target_transform_(target)

        return {"image": im, "label": target, "fm_embs": emb}


class InterPatchJEPADataset(SingleLevelHierarchicalDataset):

    def __init__(self,
                 num_context_samples: int = 1,
                 num_target_samples: int = 1,
                 **kwargs):

        super().__init__(**kwargs)
        logging.info("processing dataset instances: convert to patch dict")
        for inst in tqdm(self.instances_):
            inst["patches"] = {i["patch_name"]: i for i in inst["patches"]}
            inst["patch_names"] = sorted(list(inst["patches"].keys()))

        self.num_context_samples_ = num_context_samples
        self.num_target_samples_ = num_target_samples

    def process_read_wrap(self, curr_path, curr_inst, inst):
        im = self.process_read_im_(
            curr_path, tuple(self.tensor_shape_map[inst["name"]]["shape"]),
            curr_inst["patch_idx"])

        #imp = "@".join([curr_inst["slide_name"], curr_inst["patch_name"]])

        coord = tuple(map(int, curr_inst["patch_name"].split("-")))
        return im, coord

    def read_images_hard(self, inst: Dict):
        """Read in a list of patches, different patches and transformations"""
        raise NotImplementedError()
        im_id = np.random.permutation(inst["patch_names"])

        im_shape = np.array(
            self.tensor_shape_map[inst["name"]]["shape"][1:])[[-1, 0, 1]]
        context_images = torch.zeros(
            tuple(np.hstack([self.num_context_samples_, im_shape])))
        target_images = torch.zeros(
            tuple(
                np.hstack([
                    self.num_context_samples_, self.num_target_samples_,
                    im_shape
                ])))
        context_imps = np.empty([self.num_context_samples_], dtype=object)
        target_imps = np.empty(
            [self.num_context_samples_, self.num_target_samples_],
            dtype=object)
        target_delta = np.empty(
            [self.num_context_samples_, self.num_target_samples_, 2],
            dtype=int)
        num_context = 0
        context_permute_idx = 0
        away_round = lambda x: math.ceil(x) if x > 0 else math.floor(x)

        while num_context < self.num_context_samples_:

            # try:
            curr_inst = inst["patches"][im_id[context_permute_idx %
                                              len(im_id)]]
            curr_path = self.make_im_path(
                self.tensor_shape_map[curr_inst["slide_name"]]["path"])

            # read context
            im_ci, cxt_coord = self.process_read_wrap(curr_path, curr_inst,
                                                      inst)
            context_images[num_context, ...] = im_ci
            context_imps[num_context] = curr_inst["patch_name"]
            context_permute_idx += 1

            # read targets
            num_target = 0
            patience = 0
            while (num_target < self.num_target_samples_) and (patience < 10):
                dx = away_round(np.random.normal(scale=5))
                dy = away_round(np.random.normal(scale=5))
                new_coord = np.array(cxt_coord) + [dx, dy]
                target_patch_name = f"{new_coord[0]:04d}-{new_coord[1]:04d}"

                if (target_patch_name in inst["patches"]) and (
                        target_patch_name
                        not in target_imps[num_context][:num_target]):
                    im_tj, _ = self.process_read_wrap(
                        curr_path, inst["patches"][target_patch_name], inst)
                    target_images[num_context, num_target, ...] = im_tj
                    target_imps[num_context, num_target] = target_patch_name
                    target_delta[num_context, num_target, :] = [dx, dy]
                    num_target += 1

                patience += 1

            if num_target == self.num_target_samples_:
                num_context += 1

            #except:
            #    logging.error(
            #        "error reading context bad_file - {}".format(curr_path))

        assert self.transform_ is not None

        target_images = torch.stack([
            torch.stack([self.transform_(j) for j in i]) for i in target_images
        ])
        context_images = torch.stack(
            [self.transform_(i) for i in context_images])
        imp_maker = np.vectorize(lambda x: "@".join([inst["name"], x]))
        return {
            "context_image": context_images,
            "target_image": target_images,
            "context_path": imp_maker(context_imps).tolist(),
            "target_path": imp_maker(target_imps).tolist(),
            "target_delta": torch.tensor(target_delta)
        }

    def read_images(self, inst: Dict):
        """Read in a list of patches, different patches and transformations"""

        raise NotImplementedError()
        im_id = np.random.permutation(inst["patch_names"])

        im_shape = np.array(
            self.tensor_shape_map[inst["name"]]["shape"][1:])[[-1, 0, 1]]
        context_images = torch.zeros(
            tuple(np.hstack([self.num_context_samples_, im_shape])))
        target_images = torch.zeros(
            tuple(
                np.hstack([
                    self.num_context_samples_, self.num_target_samples_,
                    im_shape
                ])))
        context_imps = np.empty([self.num_context_samples_], dtype=object)
        target_imps = np.empty(
            [self.num_context_samples_, self.num_target_samples_],
            dtype=object)
        target_delta = np.empty(
            [self.num_context_samples_, self.num_target_samples_, 2],
            dtype=int)
        num_context = 0
        context_permute_idx = 0
        away_round = lambda x: math.ceil(x) if x > 0 else math.floor(x)

        while num_context < self.num_context_samples_:

            # try:
            curr_inst = inst["patches"][im_id[context_permute_idx %
                                              len(im_id)]]
            curr_path = self.make_im_path(
                self.tensor_shape_map[curr_inst["slide_name"]]["path"])
            #print(f"@@@ context {curr_path}")
            # read context
            im_ci, cxt_coord = self.process_read_wrap(curr_path, curr_inst,
                                                      inst)
            context_images[num_context, ...] = im_ci
            context_imps[num_context] = curr_inst["patch_name"]
            context_permute_idx += 1

            candidates = np.array([[-1, -1], [0, -1], [1, -1], [-1, 0], [1, 0],
                                   [-1, 1], [0, 1], [1, 1]])
            #candidates = np.array([[-2, -2], [-1, -2], [0, -2], [1, -2],
            #                       [2, -2], [-2, -1], [2, -1], [-2, 0], [2, 0],
            #                       [-2, 1], [2, 1], [-2, 2], [-1, 2], [0, 2],
            #                       [1, 2], [2, 2]])
            new_coord = np.array(cxt_coord) + candidates
            target_patch_name = [
                f"{nc[0]:04d}-{nc[1]:04d}" for nc in new_coord
            ]
            target_patch_name = [
                (tpn, delta)
                for tpn, delta in zip(target_patch_name, candidates)
                if tpn in inst["patches"]
            ]

            #print(f"@@@    target {target_patch_name}")

            if len(target_patch_name) >= self.num_target_samples_:
                chosen_idx = np.random.permutation(
                    len(target_patch_name))[:self.num_target_samples_]
                chosen = [target_patch_name[ci] for ci in chosen_idx]

                target_images[num_context, ...] = torch.stack([
                    self.process_read_wrap(curr_path, inst["patches"][c[0]],
                                           inst)[0] for c in chosen
                ])
                target_imps[num_context, :] = np.stack([c[0] for c in chosen])
                target_delta[num_context, :, :] = np.stack(
                    [c[1] for c in chosen])

                num_context += 1

            #except:
            #    logging.error(
            #        "error reading context bad_file - {}".format(curr_path))

        assert self.transform_ is not None

        target_images = torch.stack([
            torch.stack([self.transform_(j) for j in i]) for i in target_images
        ])
        context_images = torch.stack(
            [self.transform_(i) for i in context_images])
        imp_maker = np.vectorize(lambda x: "@".join([inst["name"], x]))
        return {
            "context_image": context_images,
            "target_image": target_images,
            "context_path": imp_maker(context_imps).tolist(),
            "target_path": imp_maker(target_imps).tolist(),
            "target_delta": torch.tensor(target_delta)
        }

    def __getitem__(self, idx: int):
        """Retrieve a list of patches, from the wholeslide specified by idx"""
        idx = idx % len(self.instances_)
        instance = self.instances_[idx]
        target = self.class_to_idx_[instance["label"]]

        im = self.read_images(instance)

        if self.target_transform_ is not None:
            target = self.target_transform_(target)

        im.update({"target": target})

        return im


class HierarchicalDataset(HierarchicalBaseDataset):
    """Patient Dataset. Treats each patient to be independent.

    Same attributes as PatientBaseDataset, with an additional one listed below.

    Attributes:
        primary_label_func_: callable to select the the primary label from a list
    """

    def __init__(self,
                 slides_file: str,
                 num_slide_samples: int = 1,
                 num_patch_samples: int = 1,
                 **args) -> None:
        """Inits the patient dataset

        Populate each attribute and walk through each patient to look for patches.
        The constructor takes in a path to a CSV slide file
        """

        super().__init__(df=pandas.read_csv(slides_file,
                                            dtype={"institution": str}),
                         **args)
        self.num_slide_samples_ = num_slide_samples
        self.num_patch_samples_ = num_patch_samples

    def read_images_slide(self, inst: List):
        raise NotImplementedError()  # fix permutation
        """Read in a list of patches, different patches and transformations"""
        im_id = np.random.permutation(np.arange(len(inst)))
        images = []
        imps_take = []

        idx = 0
        while len(images) < self.num_patch_samples_:
            curr_inst = inst[im_id[idx % len(im_id)]]
            curr_path = self.make_im_path(curr_inst["im_path"])
            try:
                images.append(self.process_read_im_(curr_path))
                imps_take.append(curr_path)
                idx += 1
            except:
                logging.error("bad_file - {}".format(curr_path))

        assert self.transform_ is not None
        xformed_im = torch.stack([
            torch.stack(
                [self.transform_(im) for _ in range(self.num_transforms_)])
            for im in images
        ])
        return xformed_im, imps_take

    def __getitem__(self, idx: int):
        """Retrieve a list of patches, from the wholeslide specified by idx"""
        idx = idx % len(self.instances_)
        patient = self.instances_[idx]

        slide_idx = np.arange(len(patient.slides))
        np.random.shuffle(slide_idx)
        num_repeat = self.num_slide_samples_ // len(patient.slides) + 1
        slide_idx = np.tile(slide_idx, num_repeat)[:self.num_slide_samples_]

        images = [
            self.read_images_slide(patient["slides"][i]["patches"])
            for i in slide_idx
        ]
        im = torch.stack([i[0] for i in images])
        imp = [i[1] for i in images]

        target = self.class_to_idx_[patient["label"]]
        if self.target_transform_ is not None:
            target = self.target_transform_(target)

        return {"image": im, "label": target, "path": [imp]}


HiDiscDataset = HierarchicalDataset
