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


def convert_epoch_to_iter(unit, steps, num_it_per_ep):
    if unit == "epoch":
        return num_it_per_ep * steps  # per epoch
    elif unit == "iter":
        return steps
    else:
        NotImplementedError("unit must be one of [epoch, iter]")


def get_optimizer_scheduler(model,
                            opt_cf: Dict,
                            num_it_per_ep: int,
                            effective_batch_size: int,
                            num_ep_total: int,
                            schd_cf: Dict = None):
    opt_str = opt_cf["which"]
    opt_params = opt_cf["params"]

    if opt_cf.get("scale_lr", False):
        assert "lr" in opt_params

        logging.info(f"scaling learn rate, was {opt_params['lr']}")
        opt_params["lr"] = opt_params["lr"] * effective_batch_size / 256
        logging.info(f"With effective batch size: {effective_batch_size}, " +
                     f"learn rate now {opt_params['lr']}")

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

    if not schd_cf:
        return optimizer, None

    # ==========================================================================

    sch_str = schd_cf["which"]
    sch_params = schd_cf["params"]
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
        if sch_params['num_warmup_steps'] < 1:
            sch_params['num_warmup_steps'] = int(
                sch_params['num_warmup_steps'] * num_ep_total * num_it_per_ep)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, sch_params['num_warmup_steps'],
            num_ep_total * num_it_per_ep, sch_params['num_cycles'])
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
