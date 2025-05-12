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
from ts2.data.cell_dataset import CellBenchDataset
from ts2.data.histology_data_module import get_num_replicate
from ts2.data.utils import get_collate_fn


class CellBenchDataModule(pl.LightningDataModule):
    possible_sets: List[str] = [
        "test_databank", "test", "pred"
    ]

    def __init__(self, config: OmegaConf):
        super().__init__()
        self.set_ = config.data.set
        self.xform_config_ = config.data.transform

        if "dataset" in config.data and config.data.dataset:
            raise NotImplementedError()

        if "test_dataset" in config.data and config.data.test_dataset:
            self.test_dset_config_ = config.data.test_dataset
            self.test_databank_dset_config_ = config.data.get("test_databank_dataset", config.data.test_dataset)
            self.test_get_train_ = config.testing.get("knn",
                                                      {}).get("do_knn", False)
        
        self.loader_config_ = config.data.get("loader")
        self.seed_ = config.infra.seed

        self.train_dataset_, self.trainval_dataset_ = None, None
        self.test_dataset_ = None

    def setup(self, stage: str):
        datasets = {
            "CellBenchDataset": CellBenchDataset,
        }
        
        transforms = {"HistologyTransform": HistologyTransform}
        if stage == "fit":
            raise NotImplementedError()
        
        if stage == "predict":

            test_xform = transforms[self.xform_config_.test.which](
                    which_set=self.set_, **self.xform_config_.test.params)
            
            test_dataset = datasets[self.test_dset_config_.which](
                transform=test_xform,
                **self.test_dset_config_.params)

            if self.test_get_train_:
                if "test_databank" in self.xform_config_:
                    testdb_xform = transforms[
                        self.xform_config_.test_databank.which
                    ](
                        which_set=self.set_,
                        **self.xform_config_.test_databank.params
                    )
                else:
                    testdb_xform = transforms[self.xform_config_.test.which](
                        which_set=self.set_, **self.xform_config_.test.params)
                               

                dbank_dataset = datasets[self.test_databank_dset_config_.which](
                    transform=testdb_xform,
                    **self.test_databank_dset_config_.params)
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
        raise NotImplementedError()

    def val_dataloader(self):
        raise NotImplementedError()

    def test_dataloader(self):
        raise NotImplementedError()

    def predict_dataloader(self):
        loader_params = OmegaConf.to_container(self.loader_config_.params.test, resolve=True)
        loader_params.update(self.get_seed_worker_and_generator(self.seed_))
        loader_params.update(OmegaConf.to_container(self.loader_config_.params.common, resolve=True))
        if loader_params.get("num_workers") == "auto":
            loader_params["num_workers"] = get_num_worker()


        if self.test_get_train_:
            if "test_databank" in self.loader_config_.params:
                db_loader_params = OmegaConf.to_container(self.loader_config_.params.test_databank, resolve=True)
                db_loader_params.update(self.get_seed_worker_and_generator(self.seed_))
                db_loader_params.update(OmegaConf.to_container(self.loader_config_.params.common, resolve=True))
                
                if db_loader_params.get("num_workers") == "auto":
                    db_loader_params["num_workers"] = get_num_worker()

            else:
                db_loader_params = OmegaConf.to_container(self.loader_config_.params.test, resolve=True)

            return [DataLoader(self.test_dataset_[0], **db_loader_params),
                DataLoader(self.test_dataset_[1], **loader_params)]
        
        else:
            return [DataLoader(ds, **loader_params) for ds in self.test_dataset_]

