from os.path import join as opj
import logging
from typing import List
import random
import copy
import numpy as np
import uuid
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, RandomSampler
from omegaconf import OmegaConf
import torchvision
import einops
from torchsrh.train.common import get_num_worker
from ts2.data.cell_meta_parser import CellCSVParser
from ts2.data.meta_parser import CachedCSVParser
from ts2.data.db_improc import instantiate_process_read
from ts2.data.transforms import HistologyTransform
from ts2.data.cell_dataset import CellDataset, CellBagDataset, CellPatchDataset, CellDatasetDINOv2
from ts2.data.histology_data_module import get_num_replicate
from ts2.data.utils import get_collate_fn


def check_collate_fn_which(loader_params, which_split):
    if cfw := loader_params.common.get("collate_fn", {}).get("which"):
        return cfw

    return loader_params[which_split].get("collate_fn", {}).get("which")


class CellDataModule(pl.LightningDataModule):
    possible_sets: List[str] = [
        "train", "trainval", "test_databank", "test", "pred"
    ]

    def __init__(self, config: OmegaConf):
        super().__init__()
        self.set_ = config.data.set
        self.parser_config_ = config.data.parser

        # run parser and save, if need be
        if self.parser_config_.which == "CellCSVParser":
            log_root = config.data.parser.cache_root

            inst_hash = uuid.uuid4().hex[:8]
            level = self.parser_config_.level
            self.instance_cache_fname_ = {
                s: opj(log_root, f"{inst_hash}_{level}_{s}")
                for s in self.possible_sets
            }

            for s in self.possible_sets:
                if s in self.parser_config_.params:
                    CellCSVParser(**self.parser_config_.params.common,
                                  **self.parser_config_.params[s])(
                                      cache_dir=self.instance_cache_fname_[s],
                                      level=level)
        else:
            self.instance_cache_fname_ = self.parser_config_.params.cached_parser_file

        self.xform_config_ = config.data.transform

        if "dataset" in config.data and config.data.dataset:
            self.dset_config_ = config.data.dataset

            combined_train_cf = {}
            combined_train_cf.update(self.dset_config_.params.common)
            combined_train_cf.update(self.dset_config_.params.train)
            num_replicate = combined_train_cf.get(
                "num_instance_self_replicate", 1)

            self.train_dset_len_ = CachedCSVParser(
                cache_dir=self.instance_cache_fname_["train"]).get_meta(
                )["instance_len"] * num_replicate

        if "test_dataset" in config.data and config.data.test_dataset:
            self.test_dset_config_ = config.data.test_dataset
            self.test_get_train_ = config.testing.get("knn",
                                                      {}).get("do_knn", False)
        self.loader_config_ = config.data.get("loader")
        self.seed_ = config.infra.seed

        self.train_dataset_, self.trainval_dataset_ = None, None
        self.test_dataset_ = None

    def setup(self, stage: str):
        datasets = {
            "CellDataset": CellDataset,
            "CellBagDataset": CellBagDataset,
            "CellPatchDataset": CellPatchDataset,
            "CellDatasetDINOv2": CellDatasetDINOv2
        }

        transforms = {"HistologyTransform": HistologyTransform}
        if stage == "fit":
            prf = instantiate_process_read(
                which=self.dset_config_.which_process_read,
                which_set=self.set_)
            train_inst, train_tsm = CachedCSVParser(
                cache_dir=self.instance_cache_fname_["train"])()

            train_xform = transforms[self.xform_config_.train.which](
                which_set=self.set_, **self.xform_config_.train.params)
            trainval_xform = transforms[self.xform_config_.trainval.which](
                which_set=self.set_, **self.xform_config_.trainval.params)

            if check_collate_fn_which(self.loader_config_.params,
                                      "train") == "SingleCellBlendedCollator":
                self.train_strong_xform = train_xform.strong_aug
                train_xform.strong_aug = torch.nn.Identity()

            if check_collate_fn_which(
                    self.loader_config_.params,
                    "trainval") == "SingleCellBlendedCollator":
                self.trainval_strong_xform = trainval_xform.strong_aug
                trainval_xform.strong_aug = torch.nn.Identity()

            self.train_dataset_ = datasets[self.dset_config_.which](
                instances=train_inst,
                tensor_shape_map=train_tsm,
                transform=train_xform,
                process_read_im=prf,
                **self.dset_config_.params.common,
                **self.dset_config_.params.train)

            trainval_inst, trainval_tsm = CachedCSVParser(
                cache_dir=self.instance_cache_fname_["trainval"])()
            self.trainval_dataset_ = datasets[self.dset_config_.which](
                instances=trainval_inst,
                tensor_shape_map=trainval_tsm,
                transform=trainval_xform,
                process_read_im=prf,
                **self.dset_config_.params.common,
                **self.dset_config_.params.trainval)

        if stage == "predict":
            prf = instantiate_process_read(
                which=self.test_dset_config_.which_process_read,
                which_set=self.set_)
            
            test_inst, test_tsm = CachedCSVParser(
                cache_dir=self.instance_cache_fname_["test"])()
            
            test_xform = transforms[self.xform_config_.test.which](
                    which_set=self.set_, **self.xform_config_.test.params)

            if check_collate_fn_which(self.loader_config_.params,
                                      "test") == "SingleCellBlendedCollator":
                self.test_strong_xform = test_xform.strong_aug
                test_xform.strong_aug = torch.nn.Identity()


            test_dataset = datasets[self.test_dset_config_.which](
                instances=test_inst,
                tensor_shape_map=test_tsm,
                transform=test_xform,
                process_read_im=prf,
                **self.test_dset_config_.params)

            if self.test_get_train_:
                dbank_inst, dbank_tsm = CachedCSVParser(
                    cache_dir=self.instance_cache_fname_["test_databank"])()
                dbank_dataset = datasets[self.test_dset_config_.which](
                    instances=dbank_inst,
                    tensor_shape_map=dbank_tsm,
                    transform=transforms[self.xform_config_.test.which](
                        which_set=self.set_, **self.xform_config_.test.params),
                    process_read_im=prf,
                    **self.test_dset_config_.params)
                self.test_dataset_ = [dbank_dataset, test_dataset]
            else:
                self.test_dataset_ = [test_dataset]

    @staticmethod
    def get_seed_worker_and_generator(seed=torch.initial_seed() % 2**32):
        """For deterministic training.

        Reference: https://pytorch.org/docs/stable/notes/randomness.html
        """

        def seed_worker(_):
            np.random.seed(seed)
            random.seed(seed)

        g = torch.Generator()
        g.manual_seed(seed)
        return {"worker_init_fn": seed_worker, "generator": g}

    def train_dataloader(self):
        loader_params = OmegaConf.to_container(
            self.loader_config_.params.train)
        loader_params.update(self.get_seed_worker_and_generator(self.seed_))
        loader_params.update(self.loader_config_.params.common)
        if loader_params.get("num_workers") == "auto":
            loader_params["num_workers"] = get_num_worker()

        if "collate_fn" in loader_params:
            if loader_params["collate_fn"][
                    "which"] == "SingleCellBlendedCollator":
                loader_params["collate_fn"]["params"].update(
                    {"strong_transforms": self.train_strong_xform})

            loader_params["collate_fn"] = get_collate_fn(
                **loader_params['collate_fn'])

        #if config["data"]["which"] == "slide_emb":
        #    train_loader_params["collate_fn"] = emb_collate_fn
        #    val_loader_params["collate_fn"] = emb_collate_fn

        logging.info(f"train loader params {loader_params}")

        return DataLoader(self.train_dataset_,
                          **loader_params)

    def val_dataloader(self):
        loader_params = OmegaConf.to_container(
            self.loader_config_.params.trainval)
        loader_params.update(self.get_seed_worker_and_generator(self.seed_))
        loader_params.update(self.loader_config_.params.common)
        if loader_params.get("num_workers") == "auto":
            loader_params["num_workers"] = get_num_worker()

        if "collate_fn" in loader_params:
            if loader_params["collate_fn"][
                    "which"] == "SingleCellBlendedCollator":
                loader_params["collate_fn"]["params"].update(
                    {"strong_transforms": self.trainval_strong_xform})

            loader_params["collate_fn"] = get_collate_fn(
                **loader_params['collate_fn'])

        if ("trainval_sampler" in self.loader_config_):
            raise ValueError(
                ("trainval_sampler no longer supported in loader config. ",
                 "implement this in your dataset"))
        logging.info(f"val loader params {loader_params}")
        return DataLoader(self.trainval_dataset_,
                          **loader_params)

    def test_dataloader(self):
        raise NotImplementedError()

    def predict_dataloader(self):
        loader_params = OmegaConf.to_container(self.loader_config_.params.test)
        loader_params.update(self.get_seed_worker_and_generator(self.seed_))
        loader_params.update(self.loader_config_.params.common)
        if loader_params.get("num_workers") == "auto":
            loader_params["num_workers"] = get_num_worker()

        if "collate_fn" in loader_params:
            if loader_params["collate_fn"]["which"] == "SingleCellBlendedCollator":
                loader_params["collate_fn"]["params"].update(
                    {"strong_transforms": self.test_strong_xform})

            loader_params["collate_fn"] = get_collate_fn(
                **loader_params['collate_fn'])
        return [DataLoader(ds, **loader_params) for ds in self.test_dataset_]


if __name__ == "__main__":
    import yaml
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format=("[%(levelname)-s|%(asctime)s|" +
                "%(filename)s:%(lineno)d|%(funcName)s] %(message)s"),
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler()])
    logging.info("Histology Data Module Debug Log")

    parser = argparse.ArgumentParser()
    parser.add_argument("-c",
                        "--config",
                        type=argparse.FileType('r'),
                        required=True,
                        help='config file for training')

    args = parser.parse_args()
    cf = OmegaConf.create(yaml.safe_load(args.config))
    #cf.data.dataset.params.train.num_instance_self_replicate = 1
    pdm = CellDataModule(cf)

    if cf.data.dataset:
        pdm.prepare_data()
        pdm.setup(stage="fit")
        tl = pdm.train_dataloader()
        vl = pdm.val_dataloader()

        from tqdm import tqdm
        for i in tqdm(range(len(tl.dataset))):
            data = tl.dataset.__getitem__(0)
            torch.save(data, f"test_data{i}.pt")
            import pdb
            pdb.set_trace()
