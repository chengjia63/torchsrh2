from datetime import datetime
import uuid
import logging
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from typing import Dict, Any
from torchsrh.train.common import (get_num_it_per_ep, setup_checkpoints)
from torchsrh.train.infra import parse_args, setup_infra_light
from ts2.lm.ssl_systems import (SimCLRSystem, SupConSystem, VICRegSystem,
                                VICRegSystemWithMask, SimSiamSystem,
                                BYOLSystem)
from torchsrh.lightning_modules.hidisc_systems import HiDiscSystem
from torchsrh.lightning_modules.xmplr_systems import ExemplarLearningSystem
from torchsrh.datasets.utils import DSU
#from opensrh.train.common import get_contrastive_dataloaders as get_opensrh_contrastive_dataloaders
from ts2.data.histology_data_module import PatchDataModule

objective_system_map = {
    "supcon": SupConSystem,
    "simclr": SimCLRSystem,
    "simsiam": SimSiamSystem,
    "byol": BYOLSystem,
    "vicreg": VICRegSystem,
    "vicreg_mask": VICRegSystemWithMask,
    "hss_simclr": HiDiscSystem,
    "hss_vicreg": HiDiscSystem,
    "hidisc_simclr": HiDiscSystem,
    "hidisc_vicreg": HiDiscSystem,
    "xmplr": ExemplarLearningSystem
}


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

    return num_it_per_ep


def main():
    # infrastructure
    def get_exp_name(cf):
        all_cmt = "sd{}_{}".format(cf["infra"]["seed"], cf["infra"]["comment"])
        time = datetime.now().strftime("%b%d-%H-%M-%S")
        return "_".join([uuid.uuid4().hex[:8], time, all_cmt])

    cf, cp_cf, artifact_dir, model_dir, exp_root = setup_infra_light(
        parse_args(), get_exp_name)
    cf = OmegaConf.create(cf)

    # setup data
    dm = PatchDataModule(config=cf)
    num_it_per_ep = get_num_it_per_train_ep(dm.train_dset_len_, cf)
    logging.info(f"actual num_it_per_ep {num_it_per_ep}")

    # setup lightning module
    objective_str = cf["training"]["objective"]["which"].lower()
    assert objective_str in objective_system_map.keys()

    lightning_module_args = {"cf": cf, "num_it_per_ep": num_it_per_ep}
    if objective_str == "xmplr":
        raise NotImplementedError()
        lightning_module_args["xmplr_loader"] = data_loaders["xmplr"]
        lightning_module_args["artifact_dir"] = artifact_dir

    con_exp = objective_system_map[objective_str](**lightning_module_args)

    if "load_backbone" in cf["training"]:
        # load lightning checkpint
        ckpt_dict = torch.load(cf["training"]["load_backbone"],
                               map_location="cpu")
        state_dict = {
            k.removeprefix("model.bb."): ckpt_dict["state_dict"][k]
            for k in ckpt_dict["state_dict"] if "model.bb" in k
        }
        con_exp.model.bb.load_state_dict(state_dict)
        logging.info("Loaded backbone")
    elif cf["training"].get("resume_checkpoint", None):
        raise NotImplementedError()
        logging.info("Loaded full lightning checkpoint")
    else:
        logging.info("Training from scratch")

    # config loggers
    logger = [
        pl.loggers.TensorBoardLogger(save_dir=exp_root, name="tb"),
        pl.loggers.CSVLogger(save_dir=exp_root, name="csv")
    ]

    # TODO: Log images
    # attach_save_image_handlers(log_contrastive_images, trainer, evaluator,
    #                            writer, cf)

    # config callbacks
    logging.info(con_exp.model)
    ckpts, ckpt_params = setup_checkpoints(cf, model_dir, num_it_per_ep)
    lr_monitor = [
        pl.callbacks.LearningRateMonitor(logging_interval="step",
                                         log_momentum=False)
    ]

    device_stat_monitor = [pl.callbacks.DeviceStatsMonitor(cpu_stats=True)]

    if cf["infra"].get("log_gpu", False):
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
        enable_progress_bar=True,
        max_epochs=cf["training"]["num_epochs"],
        callbacks=callbacks,
        default_root_dir=exp_root,
        logger=logger,
        log_every_n_steps=min(num_it_per_ep, 10),
        precision=cf["training"].get("amp", "32-true"),
        deterministic=cf["training"].get("deterministic", True),
        gradient_clip_val=0.5,
        #gradient_clip_algorithm="value",
        #profiler="simple",
        **ckpt_params)

    trainer.fit(con_exp, datamodule=dm)


if __name__ == "__main__":
    main()
