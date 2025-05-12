import os
from os.path import join as opj
from datetime import datetime
import uuid
import gzip
import logging
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from typing import Dict, Any
import json
import collections
from torch.utils.tensorboard import SummaryWriter

from dinov2.train.ssl_meta_arch import SSLMetaArch
from dinov2.train.mcm_meta_arch import MCMMetaArch
from dinov2.train.pic_meta_arch import SSLMetaArch as PICMetaArch

from dinov2.train.train import do_train
from dinov2.train.train_mcm_grad_accum import do_train as do_mcm_train
from dinov2.configs import dinov2_default_config
import dinov2.distributed as distributed
from dinov2.utils import utils as dinov2_utils
from dinov2.utils.config import apply_scaling_rules_to_cfg

from ts2.train.common import setup_checkpoints
from ts2.train.infra import (parse_args, read_process_cf, setup_infra_training,
                             setup_infra_testing, get_rank)
from ts2.data.histology_data_module import PatchDataModule
from ts2.data.cell_data_module import CellDataModule


def default_setup(seed):
    distributed.enable(overwrite=True)
    rank = distributed.get_global_rank()
    dinov2_utils.fix_random_seeds(seed + rank)


def fair_setup(cf, exp_root, model_dir):
    """
    Create configs and perform basic setups.
    """

    default_cf = OmegaConf.create(dinov2_default_config)

    if "dinov2_fair_config" in cf:
        fair_cf = OmegaConf.merge(default_cf, cf["dinov2_fair_config"])
        fair_cf.train.output_dir = model_dir
        apply_scaling_rules_to_cfg(fair_cf)
        cf.dinov2_fair_config = fair_cf

    else:
        global_cf = cf.dinov2_main_config
        global_cf.train.output_dir = model_dir
        apply_scaling_rules_to_cfg(global_cf)
        cf.dinov2_main_config = global_cf

        cf.tile_dinov2_fair_config.compute_precision = default_cf.compute_precision
        cf.patch_dinov2_fair_config.compute_precision = default_cf.compute_precision

    default_setup(cf["infra"]["seed"])

    with open(opj(exp_root, "config", "parsed_fair_config.json"), "w") as fd:
        json.dump(OmegaConf.to_container(cf), fd, sort_keys=True, indent=4)

    return cf


@torch.no_grad()
def load_uni_one_bbone(backbone, ckpt):
    backbone.cls_token.copy_(ckpt["cls_token"])
    backbone.pos_embed.copy_(ckpt["pos_embed"])
    backbone.patch_embed.load_state_dict(
        collections.OrderedDict([(k.removeprefix("patch_embed."), ckpt[k])
                                 for k in ckpt.keys()
                                 if k.startswith("patch_embed.")]))
    backbone.norm.load_state_dict(
        collections.OrderedDict([(k.removeprefix("norm."), ckpt[k])
                                 for k in ckpt.keys()
                                 if k.startswith("norm.")]))
    for b in backbone.blocks:
        expected_keys = b.state_dict().keys()
        b.load_state_dict(
            collections.OrderedDict([(k, ckpt[f"blocks.{k}"])
                                     for k in expected_keys]))


#@torch.no_grad()
#def load_dinov2_one_bbone(backbone, ckpt):
#    backbone.cls_token.copy_(ckpt["embeddings.cls_token"])
#    backbone.mask_token.copy_(ckpt["embeddings.mask_token"])
#    #backbone.pos_embed.copy_(ckpt["embeddings.position_embeddings"])
#    #backbone.patch_embed.load_state_dict(
#    #    collections.OrderedDict([(k.removeprefix("embeddings.patch_embeddings.").replace("projection", "proj"), ckpt[k])
#    #                             for k in ckpt.keys()
#    #                             if k.startswith("embeddings.patch_embeddings.")]))
#
#    backbone.norm.load_state_dict(
#        collections.OrderedDict([(k.removeprefix("layernorm."), ckpt[k])
#                                 for k in ckpt.keys()
#                                 if k.startswith("layernorm.")]))
#    for b in backbone.blocks:
#        expected_keys = b.state_dict().keys()
#        b.load_state_dict(
#            collections.OrderedDict([(k, ckpt[f"encoder.layer.{k}"])
#                                     for k in expected_keys]))


def main():
    cf = read_process_cf(parse_args())
    exp_root, model_dir = setup_infra_training(cf)
    cf = fair_setup(cf, exp_root, model_dir)

    if cf.data.set == "scsrh":
        dm = CellDataModule(config=cf)
    else:
        dm = PatchDataModule(config=cf)
    dm.setup(stage="fit")

    tb_writer = SummaryWriter(log_dir=os.path.join(exp_root, "tb"))

    logging.info("Doing training")

    assert ("load_uni_ckpt"
            not in cf.ts_wrap_config), "use cfg.student.pretrained_weights"

    if cf.ts_wrap_config.get("which", "dinov2") == "dinov2":
        if not (dm.train_dataset_.num_samples_ == 1):
            raise NotImplementedError()
            #cf.dinov2_fair_config.crops.local_crops_number *= dm.train_dataset_.num_samples_

        model = SSLMetaArch(cf.dinov2_fair_config)
        model = model.to(torch.device("cuda"))
        model.prepare_for_distributed_training()

        do_train(cf.dinov2_fair_config,
                 model,
                 dataset=dm.train_dataset_,
                 tb_writer=tb_writer,
                 resume=not cf["ts_wrap_config"].get("no_resume", True))

    #elif cf.ts_wrap_config.get("which", "dinov2") == "mcm":

    #    if not (dm.train_dataset_.num_samples_ == 1):
    #        raise NotImplementedError()


    #        #cf.tile_dinov2_fair_config.crops.local_crops_number *= dm.train_dataset_.num_samples_

    #    model = MCMMetaArch(cf.tile_dinov2_fair_config,
    #                        cf.patch_dinov2_fair_config)

    #    model = model.to(torch.device("cuda"))
    #    model.prepare_for_distributed_training()

    #    do_mcm_train(cf.dinov2_main_config,
    #                 model,
    #                 dataset=dm.train_dataset_,
    #                 tb_writer=tb_writer,
    #                 resume=not cf["ts_wrap_config"].get("no_resume", True))

    elif cf.ts_wrap_config.get("which", "dinov2") == "pic":

        if not (dm.train_dataset_.num_samples_ == 1):
            raise NotImplementedError()
            #cf.dinov2_fair_config.crops.local_crops_number *= dm.train_dataset_.num_samples_

        model = PICMetaArch(cf.dinov2_fair_config)
        model = model.to(torch.device("cuda"))
        model.prepare_for_distributed_training()

        do_train(cf.dinov2_fair_config,
                     model,
                     dataset=dm.train_dataset_,
                     tb_writer=tb_writer,
                     resume=not cf["ts_wrap_config"].get("no_resume", True))




if __name__ == "__main__":
    main()
