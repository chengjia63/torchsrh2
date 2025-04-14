import os
from os.path import join as opj
import gzip
import logging
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from typing import Dict, Any
import itertools
from torchsrh.lightning_modules.hidisc_systems import HiDiscSystem
from ts2.lm.mcm_systems import MCMSystem, CellIBOTSystem
#from ts2.lm.mcm_dinov2_systems import MCMDinov2System
from ts2.lm.cell_mil_system import CellABMILSystem
from ts2.lm.ssl_systems import (FlattenSystem, SimCLRSystem, SupConSystem,
                                VICRegSystem, IJEPASystem,
                                InterPatchJEPASystem)
from ts2.lm.dinov2_eval_system import Dinov2EvalSystem
from ts2.lm.distillation_systems import (CommitteeDistillationSystem,
                                         CRDDistillationSystem)
#from ts2.alignment.compute_foundation_embeddings import (UNIEvalSystem,
#                                                         ConchEvalSystem,
#                                                         VirchowEvalSystem,
#                                                         GigapathEvalSystem,
#                                                         PLIPEvalSystem)

from ts2.data.histology_data_module import PatchDataModule
from ts2.data.cell_data_module import CellDataModule

from ts2.train.common import setup_checkpoints
from ts2.train.infra import (parse_args, read_process_cf, setup_infra_training,
                             setup_infra_testing, get_rank)

from ts2.eval.common import get_knn_logits, load_prediction
from ts2.eval.eval_modules import do_eval
from ts2.eval.cell_mil_eval_modules import do_cell_mil_eval

lms = {
    "CellIBOTSystem": CellIBOTSystem,
    "MCMSystem": MCMSystem,
    "SupConSystem": SupConSystem,
    "SimCLRSystem": SimCLRSystem,
    #"SimSiamSystem": SimSiamSystem,
    #"BYOLSystem": BYOLSystem,
    "VICRegSystem": VICRegSystem,
    "IJEPASystem": IJEPASystem,
    "HiDiscSystem": HiDiscSystem,
    "InterPatchJEPASystem": InterPatchJEPASystem,
    "CommitteeDistillationNetwork": CommitteeDistillationSystem,
    "CRDDistillationSystem": CRDDistillationSystem,
#    "UNIEvalSystem": UNIEvalSystem,
#    "ConchEvalSystem": ConchEvalSystem,
#    "VirchowEvalSystem": VirchowEvalSystem,
#    "GigapathEvalSystem": GigapathEvalSystem,
#    "PLIPEvalSystem": PLIPEvalSystem,
    "Dinov2EvalSystem": Dinov2EvalSystem,
    "FlattenSystem": FlattenSystem,
    "CellABMILSystem": CellABMILSystem,
#    "MCMDinov2System": MCMDinov2System
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

    # setup data, possibly
    if cf.data.set == "scsrh":
        dm = CellDataModule(config=cf)
    else:
        dm = PatchDataModule(config=cf)
    dm.setup(stage="fit")

    # setup training parameters
    training_params = get_num_it_per_train_ep(len(dm.train_dataset_), cf)
    training_params.update(
        {"num_ep_total": cf["training"]["trainer_params"]["max_epochs"]})
    logging.info(f"actual num_it_per_ep {training_params}")

    # Setup lightning module
    con_exp = instantiate_lightning_module(**cf["lightning_module"],
                                           training_params=training_params)

    # Load pretrained checkpoints
    if ("load_backbone" in cf.training) and (cf.training.load_backbone
                                             is not None):
        # load lightning checkpint
        ckpt_dict = torch.load(cf.training.load_backbone.ckpt_path,
                               map_location="cpu")

        if (cf.training.load_backbone.remove_prefix is not None):
            state_dict = {
                k.removeprefix(cf.training.remove_prefix):
                ckpt_dict["state_dict"][k]
                for k in ckpt_dict["state_dict"]
                if cf.training.remove_prefix in k  #model.bb.
            }
        else:
            state_dict = ckpt_dict

        con_exp.model.bb.load_state_dict(state_dict)
        logging.info(f"Loaded backbone {cf.training.load_backbone.ckpt_path}")

    elif cf["training"].get("resume_checkpoint", None):
        raise NotImplementedError()
        logging.info("Loaded full lightning checkpoint")
    else:
        logging.info("Training from scratch")

    # config loggers
    logger = [
        pl.loggers.TensorBoardLogger(save_dir=exp_root, name="tb"),
        #pl.loggers.WandbLogger(log_model="all"),
        pl.loggers.CSVLogger(save_dir=exp_root, name="csv")
    ]

    # config callbacks
    #logging.info(con_exp.model)
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
    elif cf.lightning_module.which in {
            "UNIEvalSystem", "ConchEvalSystem", "VirchowEvalSystem",
            "GigapathEvalSystem", "PLIPEvalSystem", "FlattenSystem"
    }:
        con_exp = instantiate_lightning_module(**cf.lightning_module,
                                               training_params=None)
    elif cf.lightning_module.which == "Dinov2EvalSystem":
        con_exp = Dinov2EvalSystem(**cf.lightning_module.params)
    else:
        ckpt_path = os.path.join(cf.infra.log_dir, cf.infra.exp_name,
                                 cf.testing.ckpt_path)
        con_exp = instantiate_lightning_module_from_ckpt(ckpt=ckpt_path,
                                                         **cf.lightning_module)

    if not dm:
        if cf.data.set == "scsrh":
            dm = CellDataModule(config=cf)
        else:
            dm = PatchDataModule(config=cf)

    do_knn = cf.testing.get("knn", {}).get("do_knn", False)

    if not pred_fname:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"using device {device}")

        pred_trainer = pl.Trainer(accelerator=device,
                                  devices=1,
                                  default_root_dir=eval_root,
                                  inference_mode=True)  # deterministic=True)

        # inference
        pred_raw = pred_trainer.predict(con_exp, datamodule=dm)

        def concat_all_tensors(predictions):
            pred = {}
            for k in predictions[0].keys():
                if k == "path":
                    pred[k] = [pk for p in predictions for pk in p[k][0]]
                else:
                    pred[k] = torch.cat([p[k] for p in predictions])
            return pred

        def concat_exclude_attn(predictions):
            out = {
                "cls": list(itertools.chain(*[p["cls"] for p in predictions])),
                "logits": torch.cat([p["logits"] for p in predictions]),
                "attn":
                list(itertools.chain(*[p["attn"] for p in predictions])),
                "embs":
                list(itertools.chain(*[p["embs"] for p in predictions])),
                "label": torch.cat([p["label"] for p in predictions]),
                "path": list(itertools.chain(*[p["path"]
                                               for p in predictions]))
            }
            return out

        if isinstance(con_exp, CellABMILSystem):
            process_predictions = concat_exclude_attn
        else:
            process_predictions = concat_all_tensors

        if do_knn:
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

        if do_knn:
            pred.update({"train": load_prediction(pred_fname["train"])})

    # knn inference
    if do_knn:
        pred["val"] = get_knn_logits(cf, pred["train"], pred["val"])

        val_pred_fname = opj(pred_dir, "val_predictions.pt.gz")
        with gzip.open(val_pred_fname, "w") as fd:
            torch.save(pred["val"], fd)

    if isinstance(con_exp, CellABMILSystem):
        do_cell_mil_eval(cf, results_dir, pred, do_softmax=True)
    else:
        # metrics reporting
        do_eval(cf, results_dir, pred, do_softmax=not do_knn)


def main():
    cf = read_process_cf(parse_args())
    dm, con_exp, training_exp_root = None, None, None

    if "training" in cf:
        logging.info("Doing training")
        con_exp, dm, training_exp_root = do_training(cf)

    if ("testing" in cf) and (get_rank() == 0 or get_rank() is None):
        logging.info("Doing testing on rank 0")
        do_testing(cf, dm, con_exp, training_exp_root)


if __name__ == "__main__":
    main()
