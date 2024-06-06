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

from torchsrh.lightning_modules.hidisc_systems import HiDiscSystem
#from opensrh.train.common import get_contrastive_dataloaders as get_opensrh_contrastive_dataloaders

from ts2.lm.ssl_systems import (SimCLRSystem, SupConSystem, VICRegSystem,
                                InterPatchJEPASystem)  #,
#SimSiamSystem, BYOLSystem)
from ts2.data.histology_data_module import PatchDataModule
from ts2.train.common import setup_checkpoints
from ts2.train.infra import (parse_args, read_process_cf, setup_infra_training,
                             setup_infra_testing, get_rank)
from ts2.eval.common import get_knn_logits, load_prediction
from ts2.eval.eval_modules import do_eval

lms = {
    "SupConSystem": SupConSystem,
    "SimCLRSystem": SimCLRSystem,
    #"SimSiamSystem": SimSiamSystem,
    #"BYOLSystem": BYOLSystem,
    "VICRegSystem": VICRegSystem,
    "HiDiscSystem": HiDiscSystem,
    "InterPatchJEPASystem": InterPatchJEPASystem,
}


def instantiate_lightning_module(which: str, params, training_params):
    return lms[which](training_params=training_params, **params)


def instantiate_lightning_module_from_ckpt(which: str,
                                           ckpt: str,
                                           params,
                                           training_params=None):

    return lms[which].load_from_checkpoint(ckpt,
                                           training_params=training_params,
                                           **params)


def get_num_it_per_train_ep(train_len: int, cf: OmegaConf) -> int:
    """Calcualtes the number of iteration in each epoch.

    Args:
        train_len: length of the training set
        cf: global config

    Returns:
        num_it_per_ep: number of iteration in each epoch
    """
    if torch.cuda.is_available():
        world_size = cf.infra.SLURM_GPUS_ON_NODE * cf.infra.SLURM_JOB_NUM_NODES
    else:
        world_size = 1

    effective_batch_size = (cf.data.loader.params.train.batch_size *
                            world_size *
                            cf.training.get("accumulate_grad_batches", 1))

    num_it_per_ep = train_len // effective_batch_size

    if not cf.data.loader.params.train.drop_last:
        num_it_per_ep += ((train_len % effective_batch_size) > 0)

    return {
        "num_it_per_ep": num_it_per_ep,
        "effective_batch_size": effective_batch_size
    }


def do_training(cf):
    # setup training infra
    exp_root, model_dir = setup_infra_training(cf)

    # setup data, possibly train_
    dm = PatchDataModule(config=cf)
    training_params = get_num_it_per_train_ep(dm.train_dset_len_, cf)
    training_params.update(
        {"num_ep_total": cf["training"]["trainer_params"]["max_epochs"]})
    logging.info(f"actual num_it_per_ep {training_params}")

    con_exp = instantiate_lightning_module(**cf["lightning_module"],
                                           training_params=training_params)

    #if "load_backbone" in cf["training"]:
    #    # load lightning checkpint
    #    ckpt_dict = torch.load(cf["training"]["load_backbone"],
    #                           map_location="cpu")
    #    state_dict = {
    #        k.removeprefix("model.bb."): ckpt_dict["state_dict"][k]
    #        for k in ckpt_dict["state_dict"] if "model.bb" in k
    #    }
    #    con_exp.model.bb.load_state_dict(state_dict)
    #    logging.info("Loaded backbone")
    #elif cf["training"].get("resume_checkpoint", None):
    #    raise NotImplementedError()
    #    logging.info("Loaded full lightning checkpoint")
    #else:
    #    logging.info("Training from scratch")

    # config loggers
    logger = [
        pl.loggers.TensorBoardLogger(save_dir=exp_root, name="tb"),
        pl.loggers.CSVLogger(save_dir=exp_root, name="csv")
    ]

    # config callbacks
    logging.info(con_exp.model)
    ckpts, ckpt_params = setup_checkpoints(cf["training"]["trainval"],
                                           model_dir,
                                           training_params["num_it_per_ep"])
    lr_monitor = [
        pl.callbacks.LearningRateMonitor(logging_interval="step",
                                         log_momentum=False)
    ]

    device_stat_monitor = [pl.callbacks.DeviceStatsMonitor(cpu_stats=True)]

    if cf.infra.get("log_gpu", False):
        callbacks = ckpts + lr_monitor + device_stat_monitor
    else:
        callbacks = ckpts + lr_monitor

    ddp_stra = pl.strategies.DDPStrategy(find_unused_parameters=False,
                                         static_graph=True)

    # create trainer
    use_gpu = torch.cuda.is_available()
    trainer = pl.Trainer(
        accelerator="cuda" if use_gpu else "cpu",
        strategy=ddp_stra,
        devices=cf["infra"]["SLURM_GPUS_ON_NODE"] if use_gpu else 'auto',
        num_nodes=cf["infra"]["SLURM_JOB_NUM_NODES"],
        sync_batchnorm=True if use_gpu else False,
        callbacks=callbacks,
        default_root_dir=exp_root,
        logger=logger,
        **ckpt_params,
        **cf["training"]["trainer_params"])

    trainer.fit(con_exp, datamodule=dm)

    return con_exp, dm, exp_root


def do_testing(cf, dm, con_exp, embedded_exp_root):

    # setup training infra
    eval_root, pred_dir, results_dir, pred_fname = setup_infra_testing(
        cf, embedded_exp_root=embedded_exp_root)

    if con_exp:
        if "lightning_module" in cf.testing:
            exist_statedict = con_exp.state_dict()
            con_exp = instantiate_lightning_module(
                **cf.testing.lightning_module,
                training_params=None).load_state_dict(exist_statedict)
    else:
        ckpt_path = os.path.join(cf.infra.log_dir, cf.infra.exp_name,
                                 cf.testing.ckpt_path)
        con_exp = instantiate_lightning_module_from_ckpt(ckpt=ckpt_path,
                                                         **cf.lightning_module)

    if not dm:
        dm = PatchDataModule(config=cf)

    if not pred_fname:
        pred_trainer = pl.Trainer(
            accelerator="cuda" if torch.cuda.is_available() else "cpu",
            devices=1,
            logger=False,
            default_root_dir=eval_root,
            inference_mode=True)  # deterministic=True)

        # inference
        pred_raw = pred_trainer.predict(con_exp, datamodule=dm)

        def process_predictions(predictions):
            pred = {}
            for k in predictions[0].keys():
                if k == "path":
                    pred[k] = [pk for p in predictions for pk in p[k][0]]
                else:
                    pred[k] = torch.cat([p[k] for p in predictions])
            return pred

        if cf.testing.get("knn", {}).get("do_knn", False):
            pred = {
                "train": process_predictions(pred_raw[0]),
                "val": process_predictions(pred_raw[1])
            }
            if cf.testing.get("save_train_pred", True):
                train_pred_fname = opj(pred_dir, "train_predictions.pt.gz")
                with gzip.open(train_pred_fname, "w") as fd:
                    torch.save(pred["train"], fd)
        else:
            pred = {"val": process_predictions(pred_raw)}

            val_pred_fname = opj(pred_dir, "val_predictions.pt.gz")
            with gzip.open(val_pred_fname, "w") as fd:
                torch.save(pred["val"], fd)

    else:
        logging.info("loading predictions")
        pred = {
            "val": load_prediction(pred_fname["val"]),
        }

        if cf.testing.get("knn", {}).get("do_knn", False):
            pred.update({"train": load_prediction(pred_fname["train"])})

    # knn inference
    if cf.testing.get("knn", {}).get("do_knn", False):
        pred["val"] = get_knn_logits(cf, pred["train"], pred["val"])

        val_pred_fname = opj(pred_dir, "val_predictions.pt.gz")
        with gzip.open(val_pred_fname, "w") as fd:
            torch.save(pred["val"], fd)

    # metrics reporting
    do_eval(cf, results_dir, pred, is_knn=True)


def main():
    cf = read_process_cf(parse_args())
    dm, con_exp, training_exp_root = None, None, None

    if "training" in cf:
        logging.info("Doing training")
        con_exp, dm, training_exp_root = do_training(cf)

    if ("testing" in cf) and (get_rank() == 0):
        logging.info("Doing testing on rank 0")
        do_testing(cf, dm, con_exp, training_exp_root)


if __name__ == "__main__":
    main()
