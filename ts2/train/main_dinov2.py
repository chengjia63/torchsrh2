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

from ts2.train.common import setup_checkpoints
from ts2.train.infra import (parse_args, read_process_cf, setup_infra_training,
                             setup_infra_testing, get_rank)

from dinov2.train.ssl_meta_arch import SSLMetaArch
from dinov2.train.train import do_train
from dinov2.configs import dinov2_default_config
import dinov2.distributed as distributed
from dinov2.utils import utils as dinov2_utils
from dinov2.utils.config import apply_scaling_rules_to_cfg

from torch.utils.tensorboard import SummaryWriter


def default_setup(seed):
    distributed.enable(overwrite=True)
    rank = distributed.get_global_rank()
    dinov2_utils.fix_random_seeds(seed + rank)


def fair_setup(cf, exp_root, model_dir):
    """
    Create configs and perform basic setups.
    """

    default_cf = OmegaConf.create(dinov2_default_config)
    fair_cf = OmegaConf.merge(default_cf, cf["dinov2_fair_config"])
    fair_cf.train.output_dir = model_dir
    default_setup(cf["infra"]["seed"])
    apply_scaling_rules_to_cfg(fair_cf)

    cf.dinov2_fair_config = fair_cf
    with open(opj(exp_root, "config", "parsed_fair_config.json"), "w") as fd:
        json.dump(OmegaConf.to_container(cf), fd, sort_keys=True, indent=4)

    return cf


from ts2.data.histology_data_module import PatchDataModule


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


def main():
    cf = read_process_cf(parse_args())
    exp_root, model_dir = setup_infra_training(cf)
    cf = fair_setup(cf, exp_root, model_dir)

    dm = PatchDataModule(config=cf)
    dm.setup(stage="fit")

    tb_writer = SummaryWriter(log_dir=os.path.join(exp_root, "tb"))

    logging.info("Doing training")

    model = SSLMetaArch(cf.dinov2_fair_config)

    if cf.ts_wrap_config.get("load_uni_ckpt", {}).get("do", False):
        uni_ckpt = torch.load(cf.ts_wrap_config.load_uni_ckpt.ckpt_path,
                              map_location="cpu")
        load_uni_one_bbone(model.teacher.backbone, uni_ckpt)
        load_uni_one_bbone(model.student.backbone, uni_ckpt)
    model = model.to(torch.device("cuda"))
    model.prepare_for_distributed_training()

    do_train(cf.dinov2_fair_config,
             model,
             dataset=dm.train_dataset_,
             tb_writer=tb_writer,
             resume=not cf["ts_wrap_config"].get("no_resume", True))

    #if ("testing" in cf) and (get_rank() == 0 or get_rank() is None):
    #    logging.info("Doing testing on rank 0")
    #    do_testing(cf, dm, con_exp, training_exp_root)


if __name__ == "__main__":
    main()
