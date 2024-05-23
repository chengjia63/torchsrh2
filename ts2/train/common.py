import os
import time
import uuid
import logging
import time
import math
import random
from datetime import datetime
from typing import Callable, Optional, Dict, Any, Tuple, List
from functools import partial
import pynvml as nvml
import numpy as np
from matplotlib import pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch import optim
from torch.optim.lr_scheduler import (StepLR, LambdaLR,
                                      CosineAnnealingWarmRestarts)

import pytorch_lightning as pl
import torchmetrics

from torchsrh.models import Classifier, MLP, resnet_backbone, vit_backbone
import torchsrh.optim.lr_decay as lrd
from torchsrh.optim.cosine_schedule_warmup import (
    get_cosine_schedule_with_warmup)
from torchsrh.models.vit import get_vit_backbone


def get_exp_name(cf):
    time = datetime.now().strftime("%b%d-%H-%M-%S")
    return "-".join([uuid.uuid4().hex[:8], time, cf["infra"]["comment"]])


def get_backbone(cf: Dict) -> callable:
    resnet_arch = [
        'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
    ]

    which_bbone = cf["model"]["backbone"]["which"]
    assert which_bbone in ['vit'] + resnet_arch

    if cf["data"]["set"] == "mnist":  # 1d input
        if which_bbone == "vit": raise NotImplementedError()
        nc_in = 1
    elif cf["data"]["set"] == "srh":
        if cf["data"]["augmentations"].get(
                "srh_base_augmentation",
                "three_channels") in {"ch2_only", "ch3_only", "diff_only"}:
            if which_bbone == "vit": raise NotImplementedError()
            nc_in = 1
        else:
            nc_in = 3
    elif cf["data"]["set"] == "nzsrh":
        if cf["data"].get("srh_base_augmentation", "three_channels") in {
                "ch2_only", "ch3_only", "diff_only"
        }:
            if which_bbone == "vit": raise NotImplementedError()
            nc_in = 1 * cf["data"]["nz_filter"]["params"]["count"]
        else:
            nc_in = 3 * cf["data"]["nz_filter"]["params"]["count"]
    else:
        nc_in = 3

    if which_bbone in resnet_arch:
        train_mode = cf.get("training", {}).get("objective",
                                                {}).get("which", "")
        #eval_mode = cf.get("model", {}).get("train_alg", "")
        #use_sc_mask = ((train_mode == "vicreg_mask")
        #               or (eval_mode == "vicreg_mask"))
        #logging.info(f"model with mask {use_sc_mask}")

        return partial(
            resnet_backbone,
            num_channel_in=nc_in,
            arch=which_bbone,
            cell_mask=False,  #use_sc_mask,
            **cf["model"]["backbone"]["params"])
    elif which_bbone == "vit":
        return get_vit_backbone(**cf["model"]["backbone"]["params"])
    else:
        raise ValueError("Unsupported backbone name")


def get_optimizer_scheduler(cf: Dict, model, num_it_per_ep: int):
    opt_str = cf["training"]["optimizer"]["which"]
    opt_params = cf["training"]["optimizer"]["params"]

    if cf["training"]["optimizer"].get("scale_lr", False):
        assert "lr" in opt_params
        world_size = (cf["infra"].get("SLURM_GPUS_ON_NODE", 1) *
                      cf["infra"].get("SLURM_JOB_NUM_NODES", 1))
        train_dl_cf = cf["loader"]["direct_params"]["common"]
        train_dl_cf.update(cf["loader"]["direct_params"]["train"])
        eff_bz = train_dl_cf["batch_size"] * world_size

        logging.info("scaling learn rate, " +
                     f"was {cf['training']['optimizer']['params']['lr']}")
        opt_params["lr"] = opt_params["lr"] * eff_bz / 256
        logging.info(
            f"With effective batch size: {eff_bz}, scaling learn rate, now {opt_params['lr']}"
        )

    required_params = {
        "sgd": {"lr", "momentum", "dampening", "weight_decay", "nesterov"},
        "adam": {"lr", "betas", "eps", "weight_decay", "amsgrad"},
        "adamw": {"lr", "betas", "eps", "weight_decay", "amsgrad"},
        "adamw_ld": {"lr", "weight_decay", "layer_decay"}
    }

    if opt_str == "sgd":
        if opt_params["momentum"] == 0:
            for _ in range(99):
                logging.warning(
                    "SGD with no momentum. Are you sure this is what you want?"
                )
            time.sleep(60)
        opt_func = optim.SGD
    elif opt_str == "adam":
        opt_func = optim.Adam
    elif opt_str in {"adamw", "adamw_ld"}:
        opt_func = optim.AdamW
    else:
        raise ValueError(
            "Optimizer must be one of [sgd, adam, adamw, adamw_ld]")

    #assert opt_params.keys() == required_params[opt_str]
    if opt_str == "adamw_ld":
        param_groups = lrd.param_groups_lrd(
            model,
            opt_params["weight_decay"],
            no_weight_decay_list=model.no_weight_decay(),
            layer_decay=opt_params["layer_decay"])
        optimizer = opt_func(param_groups, lr=opt_params["lr"])
    else:
        optimizer = opt_func(
            filter(lambda p: p.requires_grad, model.parameters()),
            **opt_params)

    if "scheduler" not in cf["training"]:
        return optimizer, None

    # ==========================================================================

    sch_str = cf["training"]["scheduler"]["which"]
    sch_params = cf["training"]["scheduler"]["params"]
    required_params = {
        "step_lr": {"step_size", "step_unit", "gamma"},
        "cos_warm_restart": {"t0", "t0_unit", "t_mult", "eta_min"},
        "cos_linear_warmup": {"num_warmup_steps", "num_cycles"}
    }
    assert sch_params.keys() == required_params[sch_str]
    if sch_str == "step_lr":
        step_size = convert_epoch_to_iter(sch_params['step_unit'],
                                          sch_params['step_size'],
                                          num_it_per_ep)
        logging.info("step lr scheduler step size {}".format(step_size))
        scheduler = StepLR(optimizer,
                           step_size=step_size,
                           gamma=sch_params["gamma"])
    elif sch_str == "cos_linear_warmup":
        num_epochs = cf['training']['num_epochs']
        if sch_params['num_warmup_steps'] < 1:
            sch_params['num_warmup_steps'] = int(
                sch_params['num_warmup_steps'] * num_epochs * num_it_per_ep)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, sch_params['num_warmup_steps'],
            num_epochs * num_it_per_ep, sch_params['num_cycles'])
    elif sch_str == "cos_warm_restart":
        t0 = convert_epoch_to_iter(sch_params['t0_unit'], sch_params['t0'],
                                   num_it_per_ep)
        logging.info("cos_warm_restart lr scheduler t0 {}".format(t0))
        scheduler = CosineAnnealingWarmRestarts(optimizer,
                                                T_0=t0,
                                                T_mult=sch_params["t_mult"],
                                                eta_min=sch_params["eta_min"])
    else:
        raise NotImplementedError(
            "Scheduler must be one of [step_lr, cos_warm_restart]")

    return optimizer, scheduler


def convert_epoch_to_iter(unit, steps, num_it_per_ep):
    if unit == "epoch":
        return num_it_per_ep * steps  # per epoch
    elif unit == "iter":
        return steps
    else:
        NotImplementedError("unit must be one of [epoch, iter]")


class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


def plot_grad_flow(named_parameters, out_name):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.detach().cpu().abs().mean())

    plt.figure()
    plt.bar(layers, ave_grads, alpha=0.3, color="b")
    #plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation=20)
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.savefig(out_name)
    plt.close()


def log_gpu_worker(writer, sleep_sec: int = 5) -> None:
    try:
        nvml.nvmlInit()
        ng = nvml.nvmlDeviceGetCount()
    except nvml.NVMLError as error:
        logging.error("GPU logging error")
        logging.error(error)
        return

    while True:
        start = time.time()
        try:
            temp = {}
            mem = {}
            util = {}

            for i in range(ng):
                handle = nvml.nvmlDeviceGetHandleByIndex(i)

                temp[str(i)] = nvml.nvmlDeviceGetTemperature(
                    handle, nvml.NVML_TEMPERATURE_GPU)
                mem[str(i)] = nvml.nvmlDeviceGetMemoryInfo(
                    handle).used / 1e9  # in GB
                util[str(i)] = nvml.nvmlDeviceGetUtilizationRates(handle).gpu

            writer.add_scalars("gpu/temp".format(i), temp)
            writer.add_scalars("gpu/mem".format(i), mem)
            writer.add_scalars("gpu/util".format(i), util)

        except nvml.NVMLError as error:
            logging.error("GPU logging error")
            logging.error(error)

        duration = time.time() - start
        time.sleep(max(sleep_sec - duration, 0))


def setup_checkpoints(
    cf: Dict[str, Any], model_dir: str, num_it_per_ep: int
) -> Tuple[List[pl.callbacks.ModelCheckpoint], pl.Trainer]:

    ckpt_callable = partial(pl.callbacks.ModelCheckpoint,
                            dirpath=model_dir,
                            save_top_k=-1,
                            auto_insert_metric_name=False)
    ckpt_fmt = "ckpt-epoch{epoch}-step{step}-loss{val/sum_loss_manualepoch:.2f}"
    endofepoch_ckpt_fmt = "ckpt-endofepoch{epoch}-step{step}-loss{val/sum_loss_manualepoch:.2f}"

    if cf["freq"]["unit"] == "epoch":
        if cf["freq"]["interval"] < 1:
            iter_freq = cf["freq"]["interval"] * num_it_per_ep
            if not iter_freq.is_integer():
                logging.warning("epoch not evenly divisible")
            ckpts = [
                ckpt_callable(every_n_train_steps=int(iter_freq),
                              filename=ckpt_fmt),
                ckpt_callable(every_n_epochs=1,
                              filename=endofepoch_ckpt_fmt,
                              save_on_train_epoch_end=True)
            ]
            trainer_ckpt_params = {
                "val_check_interval": int(iter_freq),
                "check_val_every_n_epoch": 1
            }
        else:
            ckpts = [
                ckpt_callable(every_n_epochs=cf["freq"]["interval"],
                              filename=ckpt_fmt)
            ]
            trainer_ckpt_params = {
                "check_val_every_n_epoch": cf["freq"]["interval"]
            }

    elif cf["freq"]["unit"] == "iter":
        assert float(cf["freq"]["interval"]).is_integer()
        ckpts = [
            ckpt_callable(every_n_train_steps=cf["freq"]["interval"],
                          filename=ckpt_fmt)
        ]
        trainer_ckpt_params = {
            "val_check_interval": cf["freq"]["interval"],
            "check_val_every_n_epoch": None
        }
    else:
        raise ValueError()

    return ckpts, trainer_ckpt_params


def get_num_worker():
    """Estimate number of cpu workers."""
    try:
        num_worker = len(os.sched_getaffinity(0))
    except Exception:
        num_worker = os.cpu_count()

    if num_worker > 1:
        return num_worker - 1
    else:
        return torch.cuda.device_count() * 4


def get_num_it_per_ep(train_loader, cf=None):
    if cf:
        numgpu = (cf["infra"]["SLURM_GPUS_ON_NODE"] *
                  cf["infra"]["SLURM_JOB_NUM_NODES"])
        logging.info(f"num gpus total: {numgpu}")
    else:
        numgpu = torch.cuda.device_count()
        logging.warning(f"num gpus (fallback) total: {numgpu}")

    numgpu = max(1, numgpu)
    return len(train_loader) // numgpu


def get_seed_worker_and_generator(seed=torch.initial_seed() % 2**32):

    def seed_worker(worker_id):
        worker_seed = seed
        np.random.seed(seed)
        random.seed(seed)

    g = torch.Generator()
    g.manual_seed(seed)
    return seed_worker, g
