import json
import logging
import math
import os
import subprocess
import uuid
from datetime import datetime
from os.path import join as opj

from hydra.core.hydra_config import HydraConfig
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities.rank_zero import rank_zero_only


def make_ts3_run_dir(
    log_dir: str, exp_name: str, run_name: str = "", tune_name: str = ""
) -> str:
    array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID", "local")
    array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "")
    timestamp = datetime.now().strftime("%b%d-%H-%M-%S")
    run_id = uuid.uuid4().hex[:8]
    run_dir = "_".join(
        [f"s{array_job_id}", str(array_task_id), run_name, tune_name, timestamp, run_id]
    )
    if get_rank():
        return opj(log_dir, exp_name, "high_rank", run_dir)
    else:
        return opj(log_dir, exp_name, run_dir)


def make_ts3_eval_dir(checkpoint_path: str, run_name: str = "") -> str:
    checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    train_dir = os.path.dirname(checkpoint_dir)
    if os.path.basename(checkpoint_dir) != "models":
        train_dir = checkpoint_dir

    timestamp = datetime.now().strftime("%b%d-%H-%M-%S")
    run_id = uuid.uuid4().hex[:8]
    eval_name = "_".join(filter(None, [run_name, timestamp, run_id]))
    return opj(train_dir, "evals", eval_name)


def register_resolvers() -> None:
    OmegaConf.register_new_resolver("ts3_run_dir", make_ts3_run_dir, replace=True)
    OmegaConf.register_new_resolver("ts3_eval_dir", make_ts3_eval_dir, replace=True)
    OmegaConf.register_new_resolver(
        "int_mul",
        lambda *values: math.prod(int(value) for value in values),
        replace=True,
    )
    OmegaConf.register_new_resolver(
        "int_div",
        lambda value, divisor: int(value) // int(divisor),
        replace=True,
    )


def prepare_config(cf: DictConfig) -> DictConfig:
    OmegaConf.set_struct(cf, False)
    cf.infra.SLURM_GPUS_ON_NODE = int(
        os.environ.get("SLURM_GPUS_ON_NODE", torch.cuda.device_count())
    )
    cf.infra.SLURM_JOB_NUM_NODES = int(os.environ.get("SLURM_JOB_NUM_NODES", 1))
    cf.infra.rank = get_rank()
    cf.infra.world_size = (
        cf.infra.SLURM_GPUS_ON_NODE * cf.infra.SLURM_JOB_NUM_NODES
        if torch.cuda.is_available()
        else 1
    )
    return cf


def get_rank() -> int:
    return int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))


def setup_output_dirs():
    exp_root = HydraConfig.get().runtime.output_dir
    model_dir = opj(exp_root, "models")
    config_dir = opj(exp_root, "config")
    code_dir = opj(exp_root, "code")

    create_artifact_dirs([model_dir, config_dir, code_dir])

    return exp_root, model_dir, config_dir, code_dir


def setup_eval_output_dirs():
    exp_root = HydraConfig.get().runtime.output_dir
    os.makedirs(exp_root, exist_ok=True)
    return exp_root


@rank_zero_only
def create_artifact_dirs(dir_names) -> None:
    for dir_name in dir_names:
        os.makedirs(dir_name, exist_ok=False)


def config_loggers(exp_root: str) -> None:
    logging_format = (
        "[%(levelname)-s|%(asctime)s|%(name)s|"
        "%(filename)s:%(lineno)d|%(funcName)s] %(message)s"
    )
    os.makedirs(exp_root, exist_ok=True)
    handlers = [
        logging.FileHandler(opj(exp_root, "train.log")),
        logging.StreamHandler(),
    ]

    logging.basicConfig(
        level=logging.INFO,
        format=logging_format,
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )
    logging.info("Exp root %s", exp_root)


def setup_inference_infra(cf: DictConfig) -> None:
    pl.seed_everything(cf.infra.seed, workers=True)
    torch.set_float32_matmul_precision(
        cf.infra.get("float32_matmul_precision", "medium")
    )
    exp_root = setup_eval_output_dirs()
    config_loggers(exp_root)
    return exp_root


@rank_zero_only
def save_config_and_env(cf: DictConfig, config_dir: str) -> None:
    with open(opj(config_dir, "env.json"), "w", encoding="utf-8") as fd:
        json.dump(dict(os.environ), fd, sort_keys=True, indent=4)
    with open(opj(config_dir, "parsed_config.yaml"), "w", encoding="utf-8") as fd:
        fd.write(OmegaConf.to_yaml(cf, resolve=True))
    with open(opj(config_dir, "parsed_config.json"), "w", encoding="utf-8") as fd:
        json.dump(
            OmegaConf.to_container(cf, resolve=True), fd, sort_keys=True, indent=4
        )


@rank_zero_only
def copy_code_state(code_dir: str) -> None:
    commands = {
        "head.txt": ["git", "rev-parse", "HEAD"],
        "git-status.txt": ["git", "status"],
        "git-diff.txt": ["git", "diff"],
        "pip-list.txt": ["pip", "list"],
    }
    for filename, command in commands.items():
        with open(opj(code_dir, filename), "w", encoding="utf-8") as fd:
            subprocess.call(command, stdout=fd, stderr=fd)


def setup_training_infra(cf: DictConfig):
    pl.seed_everything(cf.infra.seed, workers=True)
    torch.set_float32_matmul_precision(
        cf.infra.get("float32_matmul_precision", "medium")
    )
    torch.cuda.empty_cache()

    exp_root, model_dir, config_dir, code_dir = setup_output_dirs()
    config_loggers(exp_root)
    logging.info("SLURM_JOB_NUM_NODES %s", cf.infra.SLURM_JOB_NUM_NODES)
    logging.info("SLURM_GPUS_ON_NODE %s", cf.infra.SLURM_GPUS_ON_NODE)
    save_config_and_env(cf, config_dir)
    copy_code_state(code_dir)

    return exp_root, model_dir
