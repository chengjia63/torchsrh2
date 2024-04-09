import logging
import copy
from functools import partial
from typing import Dict, Any
from abc import ABC
import gc
import psutil
import torch
from torch import nn
import pytorch_lightning as pl
import torchmetrics

from torchsrh.losses.supcon import SupConLoss
from torchsrh.losses.vicreg import GeneralVICRegLoss, FastGeneralVICRegLoss
from torchsrh.train.common import get_backbone
from torchsrh.models import (MLP, ContrastiveLearningNetwork, VICRegNetwork,
                             SimSiamNetwork)
from torchsrh.models.cnn import VICRegNetworkWithMask, EvalNetwork
from ts2.optim.utils import get_optimizer_scheduler

from memory_profiler import profile


class EvalSystem(pl.LightningModule):

    def __init__(self, cf: Dict[str, Any], **kwargs):
        super().__init__()
        self.cf_ = cf
        bb = get_backbone(cf)
        self.model = EvalNetwork(bb)


#    def forward(self, data):
#        return torch.cat([self.model(x)["emb"] for x in data["image"]], dim=1)

    @torch.inference_mode()
    def predict_step(self, batch, _):
        emb = self.model(batch["image"][:, 0, ...])
        #emb = self.all_gather(emb)
        return {
            "path": batch["path"],
            "label": batch["label"],
            "embeddings": emb
        }


class ContrastiveSystem(pl.LightningModule, ABC):
    """Lightning system for contrastive learning experiments."""

    def __init__(self, cf: Dict[str, Any], num_it_per_ep: int):
        super().__init__()
        self.cf_ = cf
        self.num_it_per_ep_ = num_it_per_ep

        bb = get_backbone(cf)
        mlp = partial(MLP,
                      n_in=bb().num_out,
                      hidden_layers=cf["model"]["mlp_hidden"],
                      n_out=cf["model"]["num_embedding_out"])
        self.model = ContrastiveLearningNetwork(bb, mlp)

        self.train_loss = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()

    def forward(self, data):
        return torch.cat([self.model(x)["emb"] for x in data["image"]], dim=1)

    def training_step(self, batch, batch_idx):
        raise NotImplementedError()

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError()

    def on_train_epoch_end(self):
        train_loss = self.train_loss.compute()
        self.log("train/contrastive_manualepoch",
                 train_loss,
                 on_epoch=True,
                 sync_dist=True,
                 rank_zero_only=True)
        logging.info(f"train/contrastive_manualepoch {train_loss}")
        self.train_loss.reset()

    def on_validation_epoch_end(self):
        val_loss = self.val_loss.compute()
        self.log("val/contrastive_manualepoch",
                 val_loss,
                 on_epoch=True,
                 sync_dist=True,
                 rank_zero_only=True)
        logging.info(f"val/contrastive_manualepoch {val_loss}")
        self.val_loss.reset()

    @torch.inference_mode()
    def predict_step(self, batch, _):
        self.model.eval()
        assert batch["image"].shape[1] == 1
        emb = self.model.bb(batch["image"][:, 0, ...])
        #emb = self.all_gather(emb)
        return {
            "path": batch["path"],
            "label": batch["label"],
            "embeddings": emb
        }

    def configure_optimizers(self):
        if "training" not in self.cf_:
            return None  # if not training, no optimizer

        opt, sch = get_optimizer_scheduler(self.cf_, self.model,
                                           self.num_it_per_ep_)

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

    def on_train_epoch_end(self):
        gc.collect()

    def on_validation_epoch_end(self):
        gc.collect()

    def on_test_epoch_end(self):
        gc.collect()

    def on_predict_epoch_end(self):
        gc.collect()


class SimCLRSystem(ContrastiveSystem):
    """Lightning system for SimCLR experiment"""

    def __init__(self, cf, num_it_per_ep):
        super().__init__(cf, num_it_per_ep)
        if ("training" in cf):
            self.criterion = SupConLoss(
                **cf["training"]["objective"]["supcon_params"])

    def training_step(self, batch, _):
        pred = [
            #self.model(batch[:, i, ...])["proj"]
            #for i in range(batch.shape[1])
            self.model(batch["image"][:, i, ...])["proj"]
            for i in range(batch["image"].shape[1])
        ]
        pred = torch.cat(pred, dim=1)

        pred_gather = self.all_gather(pred, sync_grads=True)
        pred_gather = pred_gather.reshape(-1, *pred_gather.shape[-2:])

        loss = self.criterion(pred_gather)["loss"]

        bs = batch["image"][0].shape[0] * torch.distributed.get_world_size()
        #bs = batch[0].shape[0] * torch.distributed.get_world_size()
        self.log("train/contrastive",
                 loss.detach().item(),
                 on_step=True,
                 on_epoch=True,
                 batch_size=bs,
                 rank_zero_only=True)
        self.train_loss.update(loss.detach().item(), weight=bs)

        self.log("cpu/mem",
                 psutil.virtual_memory().used,
                 on_step=True,
                 on_epoch=True,
                 batch_size=1,
                 rank_zero_only=True)
        return loss

    @torch.inference_mode()
    def validation_step(self, batch, batch_idx):
        bs = batch["image"][0].shape[0] * torch.distributed.get_world_size()
        #bs = batch[0].shape[0] * torch.distributed.get_world_size()
        pred = [
            #self.model(batch[:, i, ...])["proj"]
            #for i in range(batch.shape[1])
            self.model(batch["image"][:, i, ...])["proj"]
            for i in range(batch["image"].shape[1])
        ]
        pred = torch.cat(pred, dim=1)
        pred_gather = self.all_gather(pred, sync_grads=False)
        pred_gather = pred_gather.reshape(-1, *pred_gather.shape[-2:])

        loss = self.criterion(pred_gather)["loss"].detach().item()
        self.val_loss.update(loss, weight=bs)


class SupConSystem(ContrastiveSystem):
    """Lightning system for SupCon experiment"""

    def __init__(self, cf, num_it_per_ep):
        super().__init__(cf, num_it_per_ep)
        if ("training" in cf):
            self.criterion = SupConLoss(
                **cf["training"]["objective"]["supcon_params"])

    def training_step(self, batch, batch_idx):
        pred = [
            self.model(batch["image"][:, i, ...])["proj"]
            for i in range(batch["image"].shape[1])
        ]
        pred = torch.cat(pred, dim=1)
        pred_gather = self.all_gather(pred, sync_grads=True)
        pred_gather = pred_gather.reshape(-1, *pred_gather.shape[-2:])
        label_gather = self.all_gather(batch["label"]).reshape(-1, 1)

        loss = self.criterion(pred_gather, label_gather)["loss"]
        bs = batch["image"][0].shape[0] * torch.distributed.get_world_size()
        self.log("train/contrastive",
                 loss,
                 on_step=True,
                 on_epoch=True,
                 batch_size=bs,
                 rank_zero_only=True)
        self.train_loss.update(loss, weight=bs)
        return loss

    @torch.inference_mode()
    def validation_step(self, batch, batch_idx):
        bs = batch["image"][0].shape[0] * torch.distributed.get_world_size()
        pred = [
            self.model(batch["image"][:, i, ...])["proj"]
            for i in range(batch["image"].shape[1])
        ]
        pred = torch.cat(pred, dim=1)
        pred_gather = self.all_gather(pred, sync_grads=False)
        pred_gather = pred_gather.reshape(-1, *pred_gather.shape[-2:])
        label_gather = self.all_gather(batch["label"]).reshape(-1, 1)

        loss = self.criterion(pred_gather, label_gather)["loss"]
        self.val_loss.update(loss, weight=bs)


class VICRegSystem(ContrastiveSystem):

    def __init__(self, cf: Dict[str, Any], num_it_per_ep: int):
        super().__init__(cf=cf, num_it_per_ep=num_it_per_ep)

        bb = get_backbone(cf)
        mlp = partial(MLP,
                      n_in=bb().num_out,
                      hidden_layers=cf["model"]["mlp_hidden"],
                      n_out=cf["model"]["num_embedding_out"])
        self.model = VICRegNetwork(bb, mlp)

        if "training" in cf:
            self.criterion = GeneralVICRegLoss(
                embedding_dim=cf["model"]["num_embedding_out"],
                **cf["training"]["objective"]["params"])
        else:
            self.criterion = None

        self.train_loss = torch.nn.ModuleDict({
            n: torchmetrics.MeanMetric()
            for n in GeneralVICRegLoss.get_loss_names()
        })

        self.val_loss = torch.nn.ModuleDict({
            n: torchmetrics.MeanMetric()
            for n in GeneralVICRegLoss.get_loss_names()
        })

        self.num_it_per_ep_ = num_it_per_ep

    def on_train_epoch_end(self):
        losses = {}
        for k in self.train_loss.keys():
            losses[k] = self.train_loss[k].compute()
            self.log(f"train/{k}_manualepoch",
                     losses[k],
                     on_epoch=True,
                     sync_dist=True,
                     rank_zero_only=True)
            self.train_loss[k].reset()
        logging.info(f"train/manualepoch {losses}")

    @torch.inference_mode()
    def on_validation_epoch_end(self):
        losses = {}
        for k in self.val_loss.keys():
            losses[k] = self.val_loss[k].compute()
            self.log(f"valid/{k}_manualepoch",
                     losses[k],
                     on_epoch=True,
                     sync_dist=True,
                     rank_zero_only=True)
            self.val_loss[k].reset()
        logging.info(f"valid/manualepoch {losses}")

    def training_step(self, batch, _):
        # forward pass
        emb = [
            self.model(batch["image"][:, i, ...])
            for i in range(batch["image"].shape[1])
        ]
        emb = torch.stack(emb, dim=1)

        emb_all = self.all_gather(emb, sync_grads=True)
        emb_all = emb_all.reshape(-1, *emb_all.shape[-2:])

        losses = self.criterion(emb_all)

        bs = emb_all.shape[0] * torch.distributed.get_world_size()
        for k in self.train_loss:
            self.log(f"train/{k}",
                     losses[k],
                     on_step=True,
                     on_epoch=False,
                     batch_size=bs,
                     rank_zero_only=True)
            self.train_loss[k].update(losses[k], weight=bs)

        return losses["loss"]

    @torch.inference_mode()
    def validation_step(self, batch, _):
        emb = [
            self.model(batch["image"][:, i, ...])
            for i in range(batch["image"].shape[1])
        ]
        emb = torch.stack(emb, dim=1)

        emb_all = self.all_gather(emb, sync_grads=True)
        emb_all = emb_all.reshape(-1, *emb_all.shape[-2:])

        losses = self.criterion(emb_all)

        bs = emb_all.shape[0] * torch.distributed.get_world_size()
        for k in self.val_loss.keys():
            self.val_loss[k].update(losses[k], weight=bs)


class VICRegSystemWithMask(VICRegSystem):

    def __init__(self, cf: Dict[str, Any], num_it_per_ep: int):
        super().__init__(cf=cf, num_it_per_ep=num_it_per_ep)

        bb = get_backbone(cf)
        mlp = partial(MLP,
                      n_in=bb().num_out,
                      hidden_layers=cf["model"]["mlp_hidden"],
                      n_out=cf["model"]["num_embedding_out"])
        self.model = VICRegNetworkWithMask(bb, mlp)

        if "training" in cf:
            self.criterion = GeneralVICRegLoss(
                embedding_dim=cf["model"]["num_embedding_out"],
                **cf["training"]["objective"]["params"])
        else:
            self.criterion = None

        self.train_loss = torch.nn.ModuleDict({
            n: torchmetrics.MeanMetric()
            for n in GeneralVICRegLoss.get_loss_names()
        })

        self.val_loss = torch.nn.ModuleDict({
            n: torchmetrics.MeanMetric()
            for n in GeneralVICRegLoss.get_loss_names()
        })

        self.num_it_per_ep_ = num_it_per_ep

    @torch.inference_mode()
    def predict_step(self, batch, _):
        self.model.eval()
        assert batch["image"].shape[1] == 1
        emb = self.model.bb(batch["image"][:, 0, ...], batch["mask"][:, 0,
                                                                     ...])
        return {
            "path": batch["path"],
            "label": batch["label"],
            "embeddings": emb
        }

    def training_step(self, batch, _):
        # forward pass
        emb = [
            self.model(batch["image"][:, i, ...], batch["mask"][:, i, ...])
            for i in range(batch["image"].shape[1])
        ]
        emb = torch.stack(emb, dim=1)

        emb_all = self.all_gather(emb, sync_grads=True)
        emb_all = emb_all.reshape(-1, *emb_all.shape[-2:])

        losses = self.criterion(emb_all)

        bs = emb_all.shape[0] * torch.distributed.get_world_size()
        for k in self.train_loss:
            self.log(f"train/{k}",
                     losses[k],
                     on_step=True,
                     on_epoch=False,
                     batch_size=bs,
                     rank_zero_only=True)
            self.train_loss[k].update(losses[k], weight=bs)

        return losses["loss"]

    @torch.inference_mode()
    def validation_step(self, batch, _):
        emb = [
            self.model(batch["image"][:, i, ...], batch["mask"][:, i, ...])
            for i in range(batch["image"].shape[1])
        ]
        emb = torch.stack(emb, dim=1)

        emb_all = self.all_gather(emb, sync_grads=True)
        emb_all = emb_all.reshape(-1, *emb_all.shape[-2:])

        losses = self.criterion(emb_all)

        bs = emb_all.shape[0] * torch.distributed.get_world_size()
        for k in self.val_loss.keys():
            self.val_loss[k].update(losses[k], weight=bs)


class SimSiamSystem(ContrastiveSystem):
    """SimSiam in Pytorch lightning implementation"""

    def __init__(self, cf, num_it_per_ep):
        super().__init__(cf, num_it_per_ep)
        self.criterion = nn.CosineSimilarity(dim=1)
        self.model = SimSiamNetwork(get_backbone(cf))

    def forward(self, batch):
        out1 = self.model(batch["image"][:, 0, ...])
        out2 = self.model(batch["image"][:, 1, ...])

        loss = -(self.criterion(out1['pred'], out2['proj'].detach()).mean(
        ) + self.criterion(out2['pred'], out1['proj'].detach()).mean()) * 0.5
        return loss

    def training_step(self, batch, _):
        loss = self.forward(batch)
        bs = batch["image"].shape[0]
        self.log("train/contrastive",
                 loss,
                 on_step=True,
                 on_epoch=True,
                 batch_size=bs,
                 rank_zero_only=True)
        self.train_loss.update(loss, weight=bs)
        return loss

    @torch.inference_mode()
    def validation_step(self, batch, batch_idx):
        bs = batch["image"].shape[0]
        loss = self.forward(batch)
        self.val_loss.update(loss, weight=bs)


class BYOLSystem(ContrastiveSystem):
    """BYOL in Pytorch lightning implementation"""

    def __init__(self, cf, num_it_per_ep):
        super().__init__(cf, num_it_per_ep)
        self.beta = cf["training"]["objective"]["params"]["ema_beta"]
        # temporarily use simsiam mlp, will relax to general one in the future
        self.model = SimSiamNetwork(get_backbone(cf))
        self.target_model = self._get_target_model()

    def forward(self, batch):
        online_out1 = self.model(batch["image"][:, 0, ...])
        online_out2 = self.model(batch["image"][:, 1, ...])

        with torch.no_grad():
            target_out1 = self.target_model(batch["image"][:, 0, ...])
            target_out2 = self.target_model(batch["image"][:, 1, ...])
            target_out1["proj"].detach_()
            target_out2["proj"].detach_()

        loss1 = self.criterion(online_out1["pred"],
                               target_out2["proj"].detach())
        loss2 = self.criterion(online_out2["pred"],
                               target_out1["proj"].detach())
        loss = loss1 + loss2
        return loss.mean()

    def training_step(self, batch, _):

        loss = self.forward(batch)
        bs = batch["image"].shape[0]
        self.log("train/contrastive",
                 loss,
                 on_step=True,
                 on_epoch=True,
                 batch_size=bs,
                 rank_zero_only=True)
        self.train_loss.update(loss, weight=bs)
        return loss

    @torch.inference_mode()
    def validation_step(self, batch, batch_idx):
        loss = self.forward(batch)
        bs = batch["image"].shape[0]
        self.val_loss.update(loss, weight=bs)

    def on_before_zero_grad(self, _):
        self.update_target_encoder()

    def criterion(self, x, y):
        x = nn.functional.normalize(x, dim=-1, p=2)
        y = nn.functional.normalize(y, dim=-1, p=2)
        return 2 - 2 * (x * y).sum(dim=-1)

    def _get_target_model(self):
        target_model = copy.deepcopy(self.model)
        for p in target_model.parameters():
            p.requires_grad = False
        return target_model

    def update_target_encoder(self):
        for p, pt in zip(self.model.parameters(),
                         self.target_model.parameters()):
            pt.data = self.beta * pt.data + (1 - self.beta) * p.data
