import os
import json
import random
import logging
import argparse
import operator
import threading
import subprocess
from shutil import copy2
from os.path import join as opj
from functools import partial, reduce
from typing import Tuple, Dict, TextIO, Optional, Any
from collections.abc import Iterable
import yaml
from datetime import datetime

import uuid
from omegaconf import OmegaConf
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torchsrh.train.common import log_gpu_worker

import pytorch_lightning as pl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--config',
                        type=argparse.FileType('r'),
                        required=True,
                        help='config file for training')
    args = parser.parse_args()
    return args.config


def read_process_cf(cf_fd: TextIO) -> Tuple[OmegaConf, str]:
    """Read config file and modify it for tuning."""
    if cf_fd.name.endswith(".json"):
        cf = json.load(cf_fd)
    elif cf_fd.name.endswith(".yaml") or cf_fd.name.endswith(".yml"):
        cf = yaml.load(cf_fd, Loader=yaml.FullLoader)

    env_var = dict(os.environ)

    if "SLURM_GPUS_ON_NODE" in env_var:
        cf["infra"]["SLURM_GPUS_ON_NODE"] = int(env_var["SLURM_GPUS_ON_NODE"])
    else:
        cf["infra"]["SLURM_GPUS_ON_NODE"] = torch.cuda.device_count()

    if "SLURM_JOB_NUM_NODES" in env_var:
        cf["infra"]["SLURM_JOB_NUM_NODES"] = int(
            env_var["SLURM_JOB_NUM_NODES"])
    else:
        cf["infra"]["SLURM_JOB_NUM_NODES"] = 1

    cf["infra"]["config_fname"] = cf_fd.name

    if "tune" in cf:
        if "SLURM_ARRAY_TASK_ID" in env_var:
            cf["tune"]["taskid"] = int(env_var["SLURM_ARRAY_TASK_ID"])
        else:
            cf["tune"]["taskid"] = None

        cf = modify_tune_cf(cf)

    cf = OmegaConf.create(cf)

    if "testing" in cf:
        assert "test" in cf.data.transform
        assert "test_dataset" in cf.data
        assert "test" in cf.data.loader.params

    return cf


def modify_tune_cf(cf):
    """Modify config for tuning experiments."""
    taskid = int(cf["tune"]["taskid"] or 0)
    params = list(cf["tune"]["params"].keys())
    options = [cf["tune"]["params"][p] for p in params]

    if cf["tune"]["diagonal_items"]:
        params_to_update = {p: o[taskid] for p, o in zip(params, options)}
    else:
        lengths = [len(cf["tune"]["params"][p]) for p in params]
        inds = np.unravel_index(taskid, lengths)
        params_to_update = {p: o[i] for p, o, i in zip(params, options, inds)}

    # https://stackoverflow.com/questions/14692690/access-nested-dictionary-items-via-a-list-of-keys
    for k in params_to_update:
        keys = k.split("/")
        parent_key = keys[:-1]
        final_key = keys[-1]
        val = params_to_update[k]
        reduce(operator.getitem, parent_key, cf)[final_key] = val

    return cf


def setup_infra_training(cf: OmegaConf):
    """setup folder structure and logging for training"""
    pl.seed_everything(cf["infra"]["seed"], workers=True)
    torch.set_float32_matmul_precision(cf["infra"].get(
        "float32_matmul_precision", "highest"))
    torch.cuda.empty_cache()

    exp_root, model_dir, config_dir, code_dir = setup_output_dirs(
        cf, get_exp_instance_name(cf))

    # logging
    config_loggers(exp_root)
    log_exp_device_info(cf)

    # config and environment variable
    save_config_files(cf, config_dir)
    copy_code_diff(code_dir)

    return exp_root, model_dir


def setup_infra_testing(cf: OmegaConf,
                        embedded_exp_root: Optional[str] = None):
    pl.seed_everything(cf["infra"]["seed"], workers=True)
    torch.set_float32_matmul_precision(cf["infra"].get(
        "float32_matmul_precision", "highest"))
    torch.cuda.empty_cache()

    eval_instance_name = get_exp_instance_name(cf)
    if embedded_exp_root:
        eval_instance_name += "_embeddedeval"
        eval_root = opj(embedded_exp_root, "evals", eval_instance_name)
        pred_dir = opj(eval_root, "predictions")
        results_dir = opj(eval_root, "results")

        for d in [pred_dir, results_dir]:
            os.makedirs(d)
        pred_fname = None
    else:

        (eval_root, config_dir, pred_dir, code_dir, results_dir,
         pred_fname) = setup_testing_output_dirs(cf, eval_instance_name)

        # logging
        config_loggers(eval_root)
        log_exp_device_info(cf)

        # config and environment variable
        save_config_files(cf, config_dir)
        copy_code_diff(code_dir)

    return eval_root, pred_dir, results_dir, pred_fname


def get_exp_instance_name(cf: OmegaConf):
    all_cmt = f"sd{cf['infra']['seed']}_{cf['infra']['comment']}"
    if (tune_cmt := cf.get("tune", {}).get("taskid")) is not None:
        all_cmt += f"_tune{tune_cmt}"
    time = datetime.now().strftime("%b%d-%H-%M-%S")
    return "_".join([uuid.uuid4().hex[:8], time, all_cmt])


def setup_output_dirs(cf: Dict, exp_instance_name: str):
    """Create all output directories."""
    log_root = cf.infra.log_dir
    exp_name = cf.infra.exp_name

    if get_rank():
        exp_name = os.path.join(exp_name, "high_rank")

    exp_root = os.path.join(log_root, exp_name, exp_instance_name)

    model_dir = os.path.join(exp_root, 'models')
    config_dir = os.path.join(exp_root, 'config')
    code_dir = os.path.join(exp_root, 'code')

    for dir_name in [model_dir, config_dir, code_dir]:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
    return exp_root, model_dir, config_dir, code_dir


def setup_testing_output_dirs(cf: OmegaConf, eval_instance_name: str):
    """Create all output directories."""
    log_root = cf.infra.log_dir
    exp_name = cf.infra.exp_name

    if "ckpt_path" in cf.testing:
        training_instance_name = cf.testing.ckpt_path.split("/")[0]
    else:
        training_instance_name = "imnet"

    exp_root = os.path.join(log_root, exp_name, training_instance_name,
                            "evals", eval_instance_name)

    # generate needed folders, evals will be embedded in experiment folders
    pred_dir = os.path.join(exp_root, 'predictions')
    config_dir = os.path.join(exp_root, 'config')
    code_dir = os.path.join(exp_root, 'code')
    artifact_dir = os.path.join(exp_root, 'artifacts')
    results_dir = os.path.join(exp_root, 'results')
    for dir_name in [
            pred_dir, config_dir, code_dir, artifact_dir, results_dir
    ]:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    # if there is a previously generated prediction, also return the
    # prediction filename so we don't have to predict again
    if cf.testing.get("eval_predictions", None):
        other_eval_instance_name = cf.testing["eval_predictions"]
        pred_dir = os.path.join(log_root, exp_name, training_instance_name,
                                "evals", other_eval_instance_name,
                                "predictions")
        pred_fname = {
            "train": os.path.join(pred_dir, "train_predictions.pt.gz"),
            "val": os.path.join(pred_dir, "val_predictions.pt.gz"),
        }
    else:
        pred_fname = None

    return exp_root, config_dir, pred_dir, code_dir, results_dir, pred_fname


@pl.utilities.rank_zero.rank_zero_only
def config_loggers(exp_root):
    """Config logger for the experiments
    Sets string format and where to save.
    """

    logging_format_str = "[%(levelname)-s|%(asctime)s|%(name)s|" + \
        "%(filename)s:%(lineno)d|%(funcName)s] %(message)s"
    logging.basicConfig(level=logging.INFO,
                        format=logging_format_str,
                        datefmt="%H:%M:%S",
                        handlers=[
                            logging.FileHandler(
                                os.path.join(exp_root, 'train.log')),
                            logging.StreamHandler()
                        ],
                        force=True)
    logging.info("Exp root {}".format(exp_root))

    formatter = logging.Formatter(logging_format_str, datefmt="%H:%M:%S")
    logger = logging.getLogger("pytorch_lightning.core")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(os.path.join(exp_root, 'train.log')))
    for h in logger.handlers:
        h.setFormatter(formatter)


def log_exp_device_info(cf):
    """Log some basic device info."""
    if "tune" in cf:
        if cf["tune"]["taskid"] is None:
            logging.warning("Tune status: Tune index None. Using default 0")
        else:
            logging.info(f"Tune status: Tune index {cf['tune']['taskid']}")
    else:
        logging.info("Tune status: Not a tune job")

    logging.info(f"SLURM_JOB_NUM_NODES {cf['infra']['SLURM_JOB_NUM_NODES']}")
    logging.info(f"SLURM_GPUS_ON_NODE {cf['infra']['SLURM_GPUS_ON_NODE']}")


def save_config_files(cf: OmegaConf, config_dir):
    """Attempt to make a copy of the slurm script."""
    env_var = dict(os.environ)
    with open(os.path.join(config_dir, "env.json"), "w") as fd:
        json.dump(env_var, fd, sort_keys=True, indent=4)
    with open(os.path.join(config_dir, "parsed_config.json"), "w") as fd:
        json.dump(OmegaConf.to_container(cf), fd, sort_keys=True, indent=4)

    copy2(cf["infra"]["config_fname"], dst=config_dir)

    # copy slurm script if exists
    if 'SLURM_CONF' in env_var and 'SLURM_JOB_ID' in env_var:
        slurm_script_path = env_var['SLURM_CONF'].replace(
            "conf-cache/slurm.conf",
            f"job{env_var['SLURM_JOB_ID']}/slurm_script")

        logging.debug(f"slurm script path {slurm_script_path}")

        if os.path.exists(slurm_script_path):
            try:
                copy2(slurm_script_path, dst=config_dir)
            except:
                logging.error("Cannot save slurm script.")
        else:
            logging.error("Cannot find slurm script.")


def copy_code_diff(code_dir: str) -> None:
    """Copy the git diff, hash, and pip environment info."""
    with open(os.path.join(code_dir, 'head.txt'), 'w') as fd:
        subprocess.call(['git', 'rev-parse', 'HEAD'], stdout=fd, stderr=fd)

    with open(os.path.join(code_dir, 'git-status.txt'), 'w') as fd:
        subprocess.call(['git', 'status'], stdout=fd, stderr=fd)

    with open(os.path.join(code_dir, 'git-diff.txt'), 'w') as fd:
        subprocess.call(['git', 'diff'], stdout=fd, stderr=fd)

    with open(os.path.join(code_dir, 'pip-list.txt'), 'w') as fd:
        subprocess.call(['pip', 'list'], stdout=fd, stderr=fd)


def get_rank() -> Optional[int]:
    """Return the global rank.
    
    Reference: https://github.com/Lightning-AI/lightning/blob/7a1e0e80/src/lightning_lite/utilities/rank_zero.py#L36
    """

    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    # None to differentiate whether an environment variable was set at all
    return None
