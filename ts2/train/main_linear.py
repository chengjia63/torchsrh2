import uuid
import logging
from functools import partial
from datetime import datetime
from typing import Dict, Optional, Any
import os
from os.path import join as opj
import math
from itertools import chain

import gzip

import torch
import pytorch_lightning as pl

from collections import Counter

from ts2.train.infra import (parse_args, read_process_cf, setup_infra_training,
                             setup_infra_testing, get_rank)
from ts2.train.main import get_num_it_per_train_ep

from torch.utils.data import Dataset, DataLoader
from torch.nn.modules.loss import CrossEntropyLoss

from sklearn.metrics import (roc_auc_score, f1_score, confusion_matrix,
                             accuracy_score, average_precision_score)
import numpy as np
import pandas as pd
import json

from dinov2.layers.dino_head import DINOHead
from ts2.optim.utils import get_optimizer_scheduler                                       
from ts2.train.common import setup_checkpoints


class SimpleEmbeddingDataset(Dataset):

    def __init__(self, embed_path, balance_instance_class):
        print(embed_path)
        with gzip.open(embed_path) as fd:
            self.data = torch.load(fd)
            # dict_keys(['path', 'label', 'embeddings'])

        if "/" in self.data["path"][0]:
            self.data["path"] = [
                x.split("/")[-1].split(".")[-1] for x in self.data["path"]
            ]


        if balance_instance_class:
            num_inst = len(self.data["label"])
            num_class = len(set(self.data["label"].tolist()))

            counts = dict(Counter(self.data["label"].tolist()))
            logging.info(embed_path)
            logging.info(counts)
            max_count = max([counts[c] for c in counts])

            all_path = []
            all_label = []
            all_embeddings = []

            for l in counts:
                embeddings = self.data["embeddings"][self.data["label"] ==
                                                     l, :]
                paths = [
                    i for i, lb in zip(self.data["path"], self.data["label"])
                    if lb == l
                ]

                perm = torch.randperm(counts[l])
                embeddings = embeddings[perm]
                paths = np.array(paths)[perm].tolist()

                if counts[l] == max_count:
                    all_path.append(paths)
                    all_label.append([l] * max_count)
                    all_embeddings.append(embeddings)
                else:
                    n_rep = math.ceil(max_count / counts[l])

                    all_path.append((paths * n_rep)[:max_count])
                    all_label.append([l] * max_count)
                    all_embeddings.append(
                        embeddings.repeat((n_rep, 1))[:max_count, :])

            self.data = {
                "label": torch.tensor(list(chain(*all_label))),
                "path": list(chain(*all_path)),
                "embeddings": torch.cat(all_embeddings)
            }

        self.wts = None
        self.nc = len(set(self.data["label"].tolist()))

    def __len__(self):
        return len(self.data["label"])

    def __getitem__(self, idx):
        return (self.data["embeddings"][idx, :], self.data["label"][idx])


class SimpleLinear(pl.LightningModule):

    def __init__(self, model_hyperparams,
                 loss_params,
                 pred_dir, metric_dir,
                 opt_cf: Optional[Dict] = None,
                 schd_cf: Optional[Dict] = None,
                 training_params: Optional[Dict] = None):
        
        super(SimpleLinear, self).__init__()
        self.model = DINOHead(**model_hyperparams)
        self.criterion = CrossEntropyLoss(**loss_params)

        self.opt_cf_ = opt_cf
        self.schd_cf_ = schd_cf
        self.training_params_ = training_params

        self.pred_dir_ = pred_dir
        self.metric_dir_ = metric_dir
        self.metrics_df_ = None

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        pred = self.forward(batch[0])
        loss = self.criterion(pred, batch[1])
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        pred = self.forward(batch[0])
        loss = self.criterion(pred, batch[1])
        self.log("val/loss", loss)
        self.val_outputs_[dataloader_idx].append({
                "pred": pred.softmax(dim=1).squeeze().detach().cpu(),
                "gt": batch[1].squeeze().detach().cpu()})
        return loss

    def on_validation_epoch_start(self):
        self.val_outputs_ = {0:[], 1:[]}

    def on_validation_epoch_end(self):
        val_results, test_results = self.val_outputs_[0], self.val_outputs_[1]
        val_results = {
            "pred": torch.cat([x["pred"] for x in val_results]),
            "gt": torch.cat([x["gt"] for x in val_results])
        }
        with gzip.open(opj(self.pred_dir_, f"val_predictions_{self.current_epoch}.pt.gz"), "w") as fd:
            torch.save(val_results, fd)

        test_results = {
            "pred": torch.cat([x["pred"] for x in test_results]),
            "gt": torch.cat([x["gt"] for x in test_results])
        }
        with gzip.open(opj(self.pred_dir_, f"test_predictions_{self.current_epoch}.pt.gz"), "w") as fd:
            torch.save(test_results, fd)

        val_metrics = get_metrics(np.array(val_results["gt"]),
                                  np.array(val_results["pred"]))
        test_metrics = get_metrics(np.array(test_results["gt"]),
                                   np.array(test_results["pred"]))

        #import pdb; pdb.set_trace()
        with open(opj(self.metric_dir_, f"all_metrics_{self.current_epoch}.json"), "w") as fd:
            json.dump({"val": val_metrics, "test": test_metrics}, fd)

        columns = [f"val_{n}" for n in val_metrics["metric_names"]] + [f"test_{n}" for n in test_metrics["metric_names"]]

        metrics_df = pd.DataFrame(
            np.expand_dims(np.array(val_metrics["metrics"] + test_metrics["metrics"]),0),
            columns=columns,
            index=[self.current_epoch])
        metrics_df.to_csv(opj(self.metric_dir_, f"metrics_{self.current_epoch}.csv"))

        if self.metrics_df_ is None:
            self.metrics_df_ = metrics_df
        else:
            self.metrics_df_ = pd.concat((self.metrics_df_, metrics_df))
        pd.set_option('display.max_columns', None)
        pd.set_option('display.expand_frame_repr', False)

        logging.info(f"all metrics\n{str(self.metrics_df_)}")


    def predict_step(self, batch, batch_idx):
        pred = self.forward(batch[0])
        return {
            "pred": pred.softmax(dim=1).squeeze().detach().cpu(),
            "gt": batch[1]
        }


    def configure_optimizers(self):
        if not self.training_params_:
            return None  # if not training, no optimizer

        opt, sch = get_optimizer_scheduler(self.model,
                                           opt_cf=self.opt_cf_,
                                           schd_cf=self.schd_cf_,
                                           **self.training_params_)

        if sch:
            # get learn rate scheduler
            lr_scheduler_config = {
                "scheduler": sch,
                "interval": "step",
                "frequency": 1,
                "name": "lr"
            }
            return [opt], lr_scheduler_config
        else:
            return [opt]


def get_metrics(val_labels, y_score):
    if len(np.unique(val_labels)) > 2:
        auc = roc_auc_score(val_labels,
                            y_score,
                            average='macro',
                            multi_class='ovr')
        f1 = f1_score(val_labels, y_score.argmax(axis=1), average='macro')

        matrix = confusion_matrix(val_labels, y_score.argmax(axis=1))
        mca = (matrix.diagonal() / matrix.sum(axis=1)).mean()
        auprc = average_precision_score(val_labels, y_score, average="macro")
    else:
        auc = roc_auc_score(val_labels, y_score[:, 1])
        f1 = f1_score(val_labels, y_score.argmax(axis=1), average='binary')
        matrix = confusion_matrix(val_labels, y_score.argmax(axis=1))
        mca = (matrix.diagonal() / matrix.sum(axis=1)).mean()
        auprc = average_precision_score(val_labels, y_score[:, 1])

    acc = accuracy_score(val_labels, y_score.argmax(axis=1))
    metrics = [acc, mca, f1, auc, auprc]
    metric_names = ["acc", "mca", "f1", "auroc", "map"]

    return {
        "metrics": metrics,
        "cm": matrix.tolist(),
        "metric_names": metric_names
    }


def main_bak():

    def get_exp_name(cf):

        time = datetime.now().strftime("%b%d-%H-%M-%S")
        return "_".join([uuid.uuid4().hex[:8], time, cf["infra"]["comment"]])

    cf, cp_cf, artifact_dir, model_dir, exp_root = setup_infra_light(
        parse_args(), get_exp_name)
    assert cf["data"].get("num_slide_transforms", 1) == 1

    data_loaders = {
        "train":
        DataLoader(EmbeddingDataset(**cf["data"]["train"]),
                   **cf["loader"]["direct_params"]["common"],
                   **cf["loader"]["direct_params"]["train"]),
        "val":
        DataLoader(EmbeddingDataset(**cf["data"]["val"]),
                   **cf["loader"]["direct_params"]["common"],
                   **cf["loader"]["direct_params"]["val"]),
        "test":
        DataLoader(EmbeddingDataset(**cf["data"]["test"]),
                   **cf["loader"]["direct_params"]["common"],
                   **cf["loader"]["direct_params"]["val"]),
    }
    wts = data_loaders["train"].dataset.wts
    nc = data_loaders["train"].dataset.nc

    num_it_per_ep = get_num_it_per_ep(data_loaders["train"], cf)
    logging.info(f"actual num_it_per_ep {num_it_per_ep}")

    pred_dir = opj(exp_root, "predictions")
    metric_dir = opj(exp_root, "results")

    os.makedirs(pred_dir)
    os.makedirs(metric_dir)

    exp = SimpleLinear(cf, wts, nc, num_it_per_ep, pred_dir, metric_dir)

    # config loggers
    logger = [
        pl.loggers.TensorBoardLogger(save_dir=exp_root, name="tb"),
        pl.loggers.CSVLogger(save_dir=exp_root, name="csv")
    ]

    # config callbacks
    ckpts, ckpt_params = setup_checkpoints(cf, model_dir, num_it_per_ep)
    lr_monitor = [
        pl.callbacks.LearningRateMonitor(logging_interval="step",
                                         log_momentum=False)
    ]

    device_stat_monitor = [pl.callbacks.DeviceStatsMonitor()]

    if cf["infra"].get("log_gpu", False):
        callbacks = ckpts + lr_monitor + device_stat_monitor
    else:
        callbacks = ckpts + lr_monitor

    # create trainer. NO plans for DDP, grad_accum, ever
    use_gpu = torch.cuda.is_available()
    trainer = pl.Trainer(
        accelerator="cuda" if use_gpu else "cpu",
        devices=1,
        num_nodes=1,
        #sync_batchnorm=True if use_gpu else False,
        enable_progress_bar=True,
        max_epochs=cf["training"]["num_epochs"],
        callbacks=callbacks,
        default_root_dir=exp_root,
        logger=logger,
        log_every_n_steps=min(num_it_per_ep, 10),
        num_sanity_val_steps=0,
        precision=cf["training"].get("amp", "32-true"),
        deterministic=cf["training"].get("deterministic", True),
        **ckpt_params)
    trainer.fit(exp,
                train_dataloaders=data_loaders["train"],
                val_dataloaders=[data_loaders["val"], data_loaders["test"]])

    exp.metrics_df_.to_csv(opj(metric_dir, f"metrics_epoch.csv"))



def do_training(cf):
    # setup training infra
    exp_root, model_dir = setup_infra_training(cf)

    pred_dir = opj(exp_root, "predictions")
    metric_dir = opj(exp_root, "results")

    os.makedirs(pred_dir)
    os.makedirs(metric_dir)

    # setup data, possibly
    data_loaders = {
        "train":
        DataLoader(SimpleEmbeddingDataset(**cf.data.train_dataset.params),
                   **cf.data.loader.params.common,
                   **cf.data.loader.params.train),
        "val":
        DataLoader(SimpleEmbeddingDataset(**cf.data.val_dataset.params),
                   **cf.data.loader.params.common,
                   **cf.data.loader.params.val),

    }
    if "test_dataset" in cf.data:
        data_loaders.update({"test":
        DataLoader(SimpleEmbeddingDataset(**cf.data.test_dataset.params),
                   **cf.data.loader.params.common,
                   **cf.data.loader.params.test)})

    # setup training parameters
    training_params = get_num_it_per_train_ep(len(data_loaders["train"].dataset), cf)
    training_params.update(
        {"num_ep_total": cf["training"]["trainer_params"]["max_epochs"]})
    logging.info(f"actual num_it_per_ep {training_params}")


    # Setup lightning module
    con_exp = SimpleLinear(pred_dir=pred_dir, metric_dir=metric_dir, training_params=training_params,**cf.lm)

    # config loggers
    logger = [
        pl.loggers.TensorBoardLogger(save_dir=exp_root, name="tb"),
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

    trainer.fit(con_exp,  train_dataloaders=data_loaders["train"],
                val_dataloaders=[data_loaders["val"], data_loaders["test"]])
    con_exp.metrics_df_.to_csv(opj(metric_dir, f"metrics_epoch.csv"))
    print(con_exp.metrics_df_)
    return con_exp, data_loaders, exp_root




def main():
    cf = read_process_cf(parse_args())
    dm, con_exp, training_exp_root = None, None, None

    assert "training" in cf
    assert "testing" in cf


    logging.info("Doing training")
    con_exp, dm, training_exp_root = do_training(cf)


    logging.info("Doing testing on rank 0")
    do_testing(cf, dm, con_exp, training_exp_root)


if __name__ == "__main__":
    main()
