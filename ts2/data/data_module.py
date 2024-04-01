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

from torchsrh.train.common import get_num_worker
from ts2.data.meta_parser import PatchCSVParser, CachedCSVParser
from ts2.data.slide_dataset import SingleLevelHierarchicalDataset, HierarchicalDataset
from ts2.data.db_improc import instantiate_process_read
from ts2.data.transforms import HistologyTransform


class PatchDataModule(pl.LightningDataModule):
    possible_sets: List[str] = ["train", "val", "pred"]

    def __init__(self, log_root: str, config: OmegaConf):
        super().__init__()
        self.set_ = config.data.set
        self.transforms_ = {
            k: HistologyTransform(which_set=self.set_,
                                  **config.data.transform[k])
            for k in config.data.transform
        }
        logging.info(self.transforms_)

        self.parser_config_ = config.data.parser

        self.dset_config_ = config.data.dataset
        self.train_dataset_ = self.val_dataset_ = None

        self.loader_config_ = config.data.loader
        self.seed_ = config.infra.seed

        # run parser and save, if need be
        if self.parser_config_.which == "PatchCSVParser":
            inst_hash = uuid.uuid4().hex[:8]
            level = self.parser_config_.level
            self.instance_cache_fname_ = {
                s: opj(log_root, f"{inst_hash}_{level}_{s}")
                for s in self.possible_sets
            }

            for s in self.possible_sets:
                if s in self.parser_config_.params:
                    PatchCSVParser(**self.parser_config_.params.common,
                                   **self.parser_config_.params[s])(
                                       cache_dir=self.instance_cache_fname_[s],
                                       level=level)
        else:
            self.instance_cache_fname_ = self.parser_config_.params.cached_parser_file

        self.train_dset_len_ = CachedCSVParser().get_meta(
            self.instance_cache_fname_["train"])["instance_len"]

    def setup(self, stage: str):
        prf = instantiate_process_read(which=self.set_)

        if stage == "fit":
            train_inst = CachedCSVParser()(self.instance_cache_fname_["train"])
            self.train_dataset_ = SingleLevelHierarchicalDataset(
                instances=train_inst,
                transform=self.transforms_["train"],
                process_read_im=prf,
                **self.dset_config_.params.common,
                **self.dset_config_.params.train)
            val_inst = CachedCSVParser()(self.instance_cache_fname_["val"])
            self.val_dataset_ = SingleLevelHierarchicalDataset(
                instances=val_inst,
                transform=self.transforms_["val"],
                process_read_im=prf,
                **self.dset_config_.params.common,
                **self.dset_config_.params.val)

        if stage in {"test", "predict"}:
            val_inst = CachedCSVParser()(self.instance_cache_fname_["val"])
            self.val_dataset_ = SingleLevelHierarchicalDataset(
                instances=val_inst,
                transform=self.transforms_["val"],
                process_read_im=prf,
                **self.dset_config_.params.common,
                **self.dset_config_.params.val)

    @staticmethod
    def get_seed_worker_and_generator(seed=torch.initial_seed() % 2**32):
        """For deterministic training.
        
        Reference: https://pytorch.org/docs/stable/notes/randomness.html
        """

        def seed_worker(worker_id):
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

        #if config["data"]["which"] == "slide_emb":
        #    train_loader_params["collate_fn"] = emb_collate_fn
        #    val_loader_params["collate_fn"] = emb_collate_fn

        logging.info(loader_params)
        return DataLoader(self.train_dataset_, **loader_params)

    def val_dataloader(self):
        loader_params = OmegaConf.to_container(self.loader_config_.params.val)
        loader_params.update(self.get_seed_worker_and_generator(self.seed_))
        loader_params.update(self.loader_config_.params.common)
        if loader_params.get("num_workers") == "auto":
            loader_params["num_workers"] = get_num_worker()

        if (("val_sampler" in self.loader_config_)
                and (self.loader_config_.val_sampler.num_samples > 0)):
            loader_params.update({
                "sampler":
                RandomSampler(self.val_dataset_,
                              **self.loader_config_.val_sampler)
            })
        print(loader_params)
        return DataLoader(self.val_dataset_, **loader_params)

    def test_dataloader(self):
        loader_params = OmegaConf.to_container(self.loader_config_.params.val)
        loader_params.update(self.get_seed_worker_and_generator(self.seed_))
        loader_params.update(self.loader_config_.params.common)
        if loader_params.get("num_workers") == "auto":
            loader_params["num_workers"] = get_num_worker()
        return DataLoader(self.val_dataset_, **loader_params)

    def predict_dataloader(self):
        loader_params = OmegaConf.to_container(self.loader_config_.params.val)
        loader_params.update(self.get_seed_worker_and_generator(self.seed_))
        loader_params.update(self.loader_config_.params.common)
        if loader_params.get("num_workers") == "auto":
            loader_params["num_workers"] = get_num_worker()
        return DataLoader(self.val_dataset_, **loader_params)


if __name__ == "__main__":
    params = """
    infra:
        seed: 1000
    data:
        set: he
        parser:
            which: PatchCSVParser
            level: slide
            params:
                common:
                    data_root: /nfs/umms-tocho-mr/dropbox/data/root_he_db
                    seg_model: 08dc928c
                    slide_patch_thres: null
                    use_emb: False
                    use_patch_code_as_label: True
                    primary_label_idx: 0
                train:
                    df: /nfs/turbo/umms-tocho-ns/data/data_splits/he_all/he_train.csv
                val:
                    df: /nfs/turbo/umms-tocho-ns/data/data_splits/he_all/he_all.csv
        transform:
            train:
                base_aug_params: {}
                strong_aug_config:
                    aug_list:
                    - which: random_horiz_flip
                      params: {}
                    - which: random_vert_flip
                      params: {}
                    - which: gaussian_noise
                      params: {}
                    - which: color_jitter
                      params: {}
                    - which: random_autocontrast
                      params: {}
                    - which: random_solarize
                      params:
                        threshold: 0.2
                    - which: random_sharpness
                      params:
                        sharpness_factor: 2
                    - which: gaussian_blur
                      params:
                        kernel_size: 5
                        sigma: 1
                    - which: random_erasing
                      params: {}
                    - which: random_affine
                      params:
                        degrees: 10
                        translate: [0.1, 0.3]
                    - which: random_resized_crop
                      params:
                        size: 300
                    aug_prob: 0.3
            val: ${.train}
        dataset:
            which: TODOl
            params:
                common:
                    num_transforms: 1
                    num_samples: 1
                    num_instance_self_replicate: 1
                    max_hierarchical_replicate: 1
                    balance_instance_class: False
                train: {}
                val: {}
        loader:
            params:
                common:
                    pin_memory: True
                    prefetch_factor: 10
                    num_workers: auto
                train:
                    batch_size: 512
                    drop_last: True
                    shuffle: True
                val:
                    batch_size: 128
                    drop_last: False
                    shuffle: False
            val_sampler:
              num_samples: 64                                                             
              replacement: False  
    """
    params2 = """
    infra:
        seed: 1000
    data:
        set: he
        parser:
            which: CachedCSVParser
            params:
                cached_parser_file:
                    train: ./2f00f9f2_slide_train
                    val: ./2f00f9f2_slide_val
        transform:
                train:
                    base_aug_params: {}
                    strong_aug_config:
                        aug_list:
                        - which: random_horiz_flip
                          params: {}
                        - which: random_vert_flip
                          params: {}
                        - which: gaussian_noise
                          params: {}
                        - which: color_jitter
                          params: {}
                        - which: random_autocontrast
                          params: {}
                        - which: random_solarize
                          params:
                            threshold: 0.2
                        - which: random_sharpness
                          params:
                            sharpness_factor: 2
                        - which: gaussian_blur
                          params:
                            kernel_size: 5
                            sigma: 1
                        - which: random_erasing
                          params: {}
                        - which: random_affine
                          params:
                            degrees: 10
                            translate: [0.1, 0.3]
                        - which: random_resized_crop
                          params:
                            size: 300
                        aug_prob: 0.3
                val: ${.train}
        dataset:
            which: TODO
            params:
                common:
                    num_transforms: 1
                    num_samples: 1
                    num_instance_self_replicate: 1
                    max_hierarchical_replicate: 1
                    balance_instance_class: False
                train: {}
                val: {}
        loader:
            params:
                common:
                    pin_memory: True
                    prefetch_factor: 10
                    num_workers: auto
                train:
                    batch_size: 512
                    drop_last: True
                    shuffle: True
                val:
                    batch_size: 128
                    drop_last: False
                    shuffle: False
            val_sampler:
              num_samples: 64                                                             
              replacement: False  
    """
    pdm = PatchDataModule(log_root=".", config=OmegaConf.create(params))
    pdm.prepare_data()
    pdm.setup(stage="fit")
    tl = pdm.train_dataloader()
    vl = pdm.val_dataloader()
    import pdb
    pdb.set_trace()

    #laser_noise_config: null
    #base_aug: three_channels
