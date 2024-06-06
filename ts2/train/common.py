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
