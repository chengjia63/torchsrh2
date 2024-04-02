import logging
from multiprocessing.sharedctypes import Value
from typing import Optional, List, Dict, Any
from abc import ABC
from os.path import join as opj
import pandas
import numpy as np
import torch
from torchvision import transforms

from ts2.data.db_improc import process_read_memmap
from ts2.data.balanced_dataset import BalancedDataset


class HierarchicalBaseDataset(BalancedDataset, ABC):
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
                 process_read_im: callable = process_read_memmap,
                 num_instance_self_replicate: int = 1,
                 max_hierarchical_replicate: int = 1,
                 balance_instance_class=False,
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

        self.classes_ = None
        self.class_to_idx_ = None
        self.weights_ = None

        self.num_instance_self_replicate_ = num_instance_self_replicate
        self.max_hierarchical_replicate_ = max_hierarchical_replicate
        self.num_replicates_ = -1

        self.instances_ = instances
        self.tensor_shape_map = tensor_shape_map

        assert len(self.instances_) > 0
        self.init_weights_ = self.get_weights()
        if balance_instance_class: self.replicate_balance_instances()
        self.get_weights()

    def make_im_path(self, x):
        return opj(self.data_root_, x)

    def __len__(self):
        assert self.num_replicates_ > 0
        return len(self.instances_) * self.num_replicates_


class SingleLevelHierarchicalDataset(HierarchicalBaseDataset):

    def __init__(self, num_samples: int = 1, **kwargs):

        super().__init__(**kwargs)
        self.num_samples_ = num_samples

        # hack: replicate the dataset to make sure different sampling
        #       strategy will have the same length in dataset
        self.num_replicates_ = max(
            1, (self.num_instance_self_replicate_ *
                self.max_hierarchical_replicate_ // self.num_samples_))

    def read_images(self, inst: List):
        """Read in a list of patches, different patches and transformations"""
        im_id = np.random.permutation(np.arange(len(inst["patches"])))

        images = []
        imps_take = []
        idx = 0

        while len(images) < self.num_samples_:
            curr_inst = inst["patches"][im_id[idx % len(im_id)]]
            curr_path = self.make_im_path(
                self.tensor_shape_map[curr_inst["slide_name"]]["path"])

            try:
                images.append(
                    self.process_read_im_(
                        curr_path,
                        tuple(self.tensor_shape_map[inst["name"]]["shape"]),
                        curr_inst["patch_idx"]))
                imps_take.append(curr_path)
                idx += 1
            except:
                logging.error("bad_file - {}".format(curr_path))

        assert self.transform_ is not None
        xformed_im = torch.stack([
            self.transform_(im) for _ in range(self.num_transforms_)
            for im in images
        ])
        return xformed_im, imps_take

    def __getitem__(self, idx: int):
        """Retrieve a list of patches, from the wholeslide specified by idx"""
        idx = idx % len(self.instances_)
        instance = self.instances_[idx]
        target = self.class_to_idx_[instance["label"]]
        im, imp = self.read_images(instance)

        if self.target_transform_ is not None:
            target = self.target_transform_(target)

        return {"image": im, "label": target, "path": [imp]}


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

        # hack: replicate the dataset to make sure different sampling
        #       strategy will have the same length in dataset
        self.num_replicates_ = max(
            1, (self.num_instance_self_replicate_ *
                self.max_hierarchical_replicate_) //
            (num_slide_samples * num_patch_samples * self.num_transforms_))
        logging.info(f"number of replicates {self.num_replicates_}")

    def read_images_slide(self, inst: List):
        raise NotImplementedError()
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

if __name__ == '__main__':
    raise NotImplementedError()

    from torchsrh.datasets.db_improc import get_transformations
    from torch.utils.data import DataLoader

    logging.basicConfig(
        level=logging.DEBUG,
        format=
        "[%(levelname)-s|%(asctime)s|%(filename)s:%(lineno)d|%(funcName)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler()])
    logging.info("Patch Data Debug Log")

    csv_path = "/nfs/turbo/umms-tocho/code/chengjia/torchsrh/torchsrh/train/data/srh7v1/srh7v1_toy.csv"
    data_root = "/nfs/turbo/umms-tocho/root_srh_db/"
    tx, vx = get_transformations()

    dset_params = {
        "data_root": data_root,
        "slides_file": csv_path,
        "segmentation_model": "03207B00",
        "transform": tx,
        "balance_instance_class": True,
        "num_instance_self_replicate": 2
    }

    dset = SingleLevelHierarchicalDataset(**dset_params)
    data = dset.__getitem__(10)

    dset = HierarchicalDataset(data_root=data_root,
                               slides_file=csv_path,
                               segmentation_model="03207B00",
                               transform=tx,
                               balance_instance_class=True,
                               num_slide_samples=6,
                               num_patch_samples=5,
                               num_transforms=4,
                               max_hierarchical_replicate=6 * 5 * 4)
    dl = DataLoader(dset, batch_size=7)
    batch1 = next(iter(dl))
    assert batch1["image"].shape == torch.Size([7, 6, 5, 4, 3, 300, 300])
    # (#patient(batch size)) * (#slide per patient) * (#patch per slide) * (#xform per patch) * im_size
    assert batch1["label"].shape == torch.Size([7])
    assert np.array(batch1["path"]).shape == (1, 6, 5, 7)
    import pdb; pdb.set_trace() #yapf:disable
