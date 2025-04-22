# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import argparse
import logging
import math
import os
from functools import partial

from fvcore.common.checkpoint import PeriodicCheckpointer
import torch

from dinov2.data import SamplerType, make_data_loader
from dinov2.data import collate_data_and_cast, MaskingGenerator
from dinov2.data.collate import collate_tile_patch_data_and_cast_fmi
import dinov2.distributed as distributed
from dinov2.fsdp import FSDPCheckpointer
from dinov2.logging import MetricLogger
#from dinov2.utils.config import setup
from dinov2.utils.utils import CosineScheduler

from dinov2.train.ssl_meta_arch import SSLMetaArch
from dinov2.fsdp import rankstr
from dinov2.train.train import build_optimizer, build_schedulers, apply_optim_scheduler

torch.backends.cuda.matmul.allow_tf32 = True  # PyTorch 1.12 sets this to False by default
logger = logging.getLogger("dinov2")


def do_test(cfg, model, iteration):
    tile_state_dict = model.tile.teacher.state_dict()
    patch_state_dict = model.patch.teacher.state_dict()

    if distributed.is_main_process():
        iterstring = str(iteration)
        eval_dir = os.path.join(cfg.train.output_dir, "eval", iterstring)
        os.makedirs(eval_dir, exist_ok=True)
        # save teacher checkpoint
        teacher_ckp_path = os.path.join(eval_dir, "teacher_checkpoint.pth")
        torch.save(
            {
                "tile_teacher": tile_state_dict,
                "patch_teacher": patch_state_dict
            }, teacher_ckp_path)

        
def do_train(cfg, model, dataset, tb_writer, resume=False):
    model.train()
    inputs_dtype = torch.half
    fp16_scaler = model.fp16_scaler  # for mixed precision training

    # setup optimizer

    optimizer = build_optimizer(cfg, model.get_params_groups())
    (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    ) = build_schedulers(cfg)

    # checkpointer
    checkpointer = FSDPCheckpointer(model,
                                    cfg.train.output_dir,
                                    optimizer=optimizer,
                                    save_to_disk=True)

    start_iter = checkpointer.resume_or_load(
        cfg.MODEL.WEIGHTS.replace("rank_0",
                                  rankstr()),  # input: model_xxxxxx.rank_0.pth
        resume=resume).get("iteration", -1) + 1

    #start_iter = checkpointer.resume_or_load(
    #    cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1

    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    max_iter = cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer,
        period=3 * OFFICIAL_EPOCH_LENGTH,
        max_iter=max_iter,
        max_to_keep=3,
    )

    # setup data preprocessing - Moved outside
    img_size = cfg.crops.global_crops_size
    patch_size = cfg.student.patch_size
    n_tokens = (img_size // patch_size)**2
    mask_generator = MaskingGenerator(input_size=(img_size // patch_size,
                                                  img_size // patch_size),
                                      max_num_patches=0.5 * n_tokens)

    # patch collate
    patch_img_size = cfg.crops.patch.global_crops_size
    patch_patch_size = 1  #cfg.student.patch_size
    patch_n_tokens = (patch_img_size // patch_patch_size)**2
    patch_mask_generator = MaskingGenerator(
        input_size=(patch_img_size // patch_patch_size,
                    patch_img_size // patch_patch_size),
        max_num_patches=0.5 * patch_n_tokens)

    collate_fn = partial(collate_tile_patch_data_and_cast_fmi,
                         mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
                         mask_probability=cfg.ibot.mask_sample_probability,
                         n_tokens=n_tokens,
                         mask_generator=mask_generator,
                         patch_n_tokens=n_tokens,
                         patch_mask_generator=patch_mask_generator,
                         dtype=inputs_dtype)

    # sampler_type = SamplerType.INFINITE
    sampler_type = SamplerType.SHARDED_INFINITE
    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        seed=start_iter,  # TODO: Fix this -- cfg.train.seed
        sampler_type=sampler_type,
        sampler_advance= start_iter * cfg.train.batch_size_per_gpu,
        drop_last=True,
        collate_fn=collate_fn,
    )

    # training loop

    iteration_data = start_iter
    iteration_eff = start_iter
    max_iter_data = max_iter * cfg.train.grad_accum

    logger.info("Starting training from iteration {}".format(start_iter))
    metrics_file = os.path.join(cfg.train.output_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    header = "Training"

    loss_scale = 1.0 / cfg.train.grad_accum
    for data in metric_logger.log_every(
            data_loader,
            10,
            header,
            max_iter_data,
            start_iter,
    ):

        current_batch_size = data["collated_global_crops"].shape[0] / 2
        if iteration_eff > max_iter:
            return

        # apply schedules
        teacher_temp = teacher_temp_schedule[iteration_eff]
        lr = lr_schedule[iteration_eff]
        wd = wd_schedule[iteration_eff]
        mom = momentum_schedule[iteration_eff]
        last_layer_lr = last_layer_lr_schedule[iteration_eff]
        if (iteration_data + 1) % cfg.train.grad_accum == 0:
            apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)

        # compute losses
        optimizer.zero_grad(set_to_none=True)
        loss_dict = model.forward_backward(data,
                                           teacher_temp=teacher_temp,
                                           loss_scale=loss_scale)

        if (iteration_data + 1) % cfg.train.grad_accum == 0:

            # clip gradients
            if fp16_scaler is not None:
                if cfg.optim.clip_grad:
                    fp16_scaler.unscale_(optimizer)
                    for v in model.tile.student.values():
                        v.clip_grad_norm_(cfg.optim.clip_grad)

                    for v in model.patch.student.values():
                        v.clip_grad_norm_(cfg.optim.clip_grad)

                fp16_scaler.step(optimizer)
                fp16_scaler.update()
            else:
                if cfg.optim.clip_grad:
                    for v in model.tile.student.values():
                        v.clip_grad_norm_(cfg.optim.clip_grad)
                    for v in model.patch.student.values():
                        v.clip_grad_norm_(cfg.optim.clip_grad)

                optimizer.step()

            # perform teacher EMA update
            model.update_teacher(mom)

        # logging
        if distributed.get_global_size() > 1:
            for v in loss_dict.values():
                torch.distributed.all_reduce(v)
        loss_dict_reduced = {
            k: v.item() / distributed.get_global_size()
            for k, v in loss_dict.items()
        }

        if math.isnan(sum(loss_dict_reduced.values())):
            logger.info("NaN detected")
            raise AssertionError
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        metric_logger.update(lr=lr)
        metric_logger.update(wd=wd)
        metric_logger.update(mom=mom)
        metric_logger.update(last_layer_lr=last_layer_lr)
        metric_logger.update(current_batch_size=current_batch_size)
        metric_logger.update(total_loss=losses_reduced, **loss_dict_reduced)

        if (iteration_data +
                1) % cfg.train.grad_accum == 0:  # only on update steps
            if distributed.is_main_process():
                tb_log_items = {
                    "lr": lr,
                    "wd": wd,
                    "mom": mom,
                    "last_layer_lr": last_layer_lr,
                    "current_batch_size": current_batch_size,
                    "total_loss": losses_reduced
                }
                tb_log_items.update(loss_dict_reduced)
                for k in tb_log_items:
                    tb_writer.add_scalar(k, tb_log_items[k], iteration_eff)

            # checkpointing and testing
            if cfg.evaluation.eval_period_iterations > 0 and (
                    iteration_eff +
                    1) % cfg.evaluation.eval_period_iterations == 0:
                do_test(cfg, model, f"training_{iteration_eff}")
                torch.cuda.synchronize()
            periodic_checkpointer.step(iteration_eff)

            iteration_eff += 1

        iteration_data = iteration_data + 1

    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
