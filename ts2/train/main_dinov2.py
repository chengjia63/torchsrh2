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

from ts2.train.common import setup_checkpoints
from ts2.train.infra import (parse_args, read_process_cf, setup_infra_training,
                             setup_infra_testing, get_rank)

from dinov2.train.ssl_meta_arch import SSLMetaArch
from dinov2.train.train import do_train
from dinov2.configs import dinov2_default_config
import dinov2.distributed as distributed
from dinov2.utils import utils as dinov2_utils
from dinov2.utils.config import apply_scaling_rules_to_cfg


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


def main():
    cf = read_process_cf(parse_args())
    exp_root, model_dir = setup_infra_training(cf)
    cf = fair_setup(cf, exp_root, model_dir)
    import pdb
    pdb.set_trace()
    # dset stuff
    #img_size = cfg.crops.global_crops_size
    #patch_size = cfg.student.patch_size
    #n_tokens = (img_size // patch_size)**2
    #mask_generator = MaskingGenerator(
    #    input_size=(img_size // patch_size, img_size // patch_size),
    #    max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
    #)
    #
    #data_transform = DataAugmentationDINO(
    #    cfg.crops.global_crops_scale,
    #    cfg.crops.local_crops_scale,
    #    cfg.crops.local_crops_number,
    #    global_crops_size=cfg.crops.global_crops_size,
    #    local_crops_size=cfg.crops.local_crops_size,
    #)
    #
    #collate_fn = partial(
    #    collate_data_and_cast,
    #    mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
    #    mask_probability=cfg.ibot.mask_sample_probability,
    #    n_tokens=n_tokens,
    #    mask_generator=mask_generator,
    #    dtype=inputs_dtype,
    #)

    # setup data loader

    #dataset = make_dataset(
    #    dataset_str=cfg.train.dataset_path,
    #    transform=data_transform,
    #    target_transform=lambda _: (),
    #)
    ## ========================

    logging.info("Doing training")
    model = SSLMetaArch(cf.dinov2_fair_config).to(torch.device("cuda"))
    model.prepare_for_distributed_training()

    do_train(cf.dinov2_fair_config, model, dataset=None,
             collate_fn=None)  #, resume=not cf.no_resume)

    #if ("testing" in cf) and (get_rank() == 0 or get_rank() is None):
    #    logging.info("Doing testing on rank 0")
    #    do_testing(cf, dm, con_exp, training_exp_root)


if __name__ == "__main__":
    main()
