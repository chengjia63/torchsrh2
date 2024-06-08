import logging
import itertools
import copy
from functools import partial
from typing import Dict, Any, Optional
from abc import ABC
import gc
import psutil
import torch
from torch import nn
import torch.nn.functional as F
import einops

import pytorch_lightning as pl
import torchmetrics

from torchsrh.losses.supcon import SupConLoss
from torchsrh.losses.vicreg import GeneralVICRegLoss
from ts2.models.ssl import (MLP, ContrastiveLearningNetwork, VICRegNetwork)
#SimSiamNetwork)  #, VICRegNetworkWithMask)
from ts2.models.pjepa import InterPatchJEPANetwork
from ts2.models.ijepa import IJEPANetwork, apply_masks, repeat_interleave_batch
from ts2.optim.utils import get_optimizer_scheduler

from memory_profiler import profile


class EvalBaseSystem(pl.LightningModule, ABC):
    @torch.inference_mode()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        assert batch["image"].shape[1] == 1
        emb = self.model.bb(batch["image"][:, 0, ...])
        results = {
            "path": batch["path"],
            "label": batch["label"],
            "embeddings": emb
        }

        return results

    def test_step(self, batch, batch_idx, dataloader_idx):
        raise NotImplementedError()


class ContrastiveBaseSystem(EvalBaseSystem):
    """Lightning system for contrastive learning experiments."""
    def __init__(self,
                 model_hyperparams,
                 opt_cf: Optional[Dict] = None,
                 schd_cf: Optional[Dict] = None,
                 training_params: Optional[Dict] = None):
        super().__init__()

        self.opt_cf_ = opt_cf
        self.schd_cf_ = schd_cf
        self.training_params_ = training_params

        self.model = ContrastiveLearningNetwork(**model_hyperparams)
        self.criterion = None

        self.train_loss = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()
        self.test_output = []

    def forward(self, data):
        return torch.cat([self.model(x)["emb"] for x in data["image"]], dim=1)

    def on_train_epoch_end(self):
        train_loss = self.train_loss.compute()
        self.log("train/contrastive_manualepoch",
                 train_loss,
                 on_epoch=True,
                 sync_dist=True,
                 rank_zero_only=True)
        logging.info(f"train/contrastive_manualepoch {train_loss}")
        self.train_loss.reset()
        gc.collect()

    def on_validation_epoch_end(self):
        val_loss = self.val_loss.compute()
        self.log("val/contrastive_manualepoch",
                 val_loss,
                 on_epoch=True,
                 sync_dist=True,
                 rank_zero_only=True)
        logging.info(f"val/contrastive_manualepoch {val_loss}")
        self.val_loss.reset()
        gc.collect()

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


class SimCLRSystem(ContrastiveBaseSystem):
    """Lightning system for SimCLR experiment"""
    def __init__(self, loss_params, **kwargs):
        super().__init__(**kwargs)
        if self.opt_cf_:
            self.criterion = SupConLoss(**loss_params)

    def training_step(self, batch, _):
        pred = [
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


class SupConSystem(ContrastiveBaseSystem):
    """Lightning system for SupCon experiment"""
    def __init__(self, loss_params, **kwargs):
        super().__init__(**kwargs)

        if self.opt_cf_: self.criterion = SupConLoss(**loss_params)

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


class VICRegSystem(ContrastiveBaseSystem):
    def __init__(self, loss_params, **kwargs):
        super().__init__(**kwargs)

        self.model = VICRegNetwork(**kwargs['model_hyperparams'])
        if self.opt_cf_:
            self.criterion = GeneralVICRegLoss(
                embedding_dim=self.model.bb.num_out, **loss_params)

        self.train_loss = torch.nn.ModuleDict({
            n: torchmetrics.MeanMetric()
            for n in GeneralVICRegLoss.get_loss_names()
        })

        self.val_loss = torch.nn.ModuleDict({
            n: torchmetrics.MeanMetric()
            for n in GeneralVICRegLoss.get_loss_names()
        })

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


class IJEPASystem(EvalBaseSystem):
    def __init__(self,
                 model_hyperparams,
                 loss_params: Optional[Dict] = None,
                 ema_beta: Optional[Dict] = None,
                 opt_cf: Optional[Dict] = None,
                 schd_cf: Optional[Dict] = None,
                 training_params: Optional[Dict] = None):
        super().__init__()

        self.beta = ema_beta
        self.opt_cf_ = opt_cf
        self.schd_cf_ = schd_cf
        self.training_params_ = training_params

        self.model = IJEPANetwork(**model_hyperparams)

        if training_params:
            self.criterion = torch.nn.SmoothL1Loss(**loss_params)
        else:
            self.criterion = None

        self.train_loss = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()
        self.test_output = []

    def training_step(self, batch, _):
        imgs, masks_enc, masks_pred = batch
        imgs = imgs['image'].squeeze()

        def forward_target():
            with torch.no_grad():
                h = self.model.target_encoder(imgs)
                h = F.layer_norm(h, (h.size(-1), ))
                B = len(h)
                # -- create targets (masked regions of h)
                h = apply_masks(h, masks_pred)
                h = repeat_interleave_batch(h, B, repeat=len(masks_enc))
                return h

        def forward_context():
            z = self.model.encoder(imgs, masks_enc)
            z = self.model.predictor(z, masks_enc, masks_pred)
            return z

        def loss_fn(z, h):
            loss = self.criterion(z, h)
            # TODO
            # loss = AllReduce.apply(loss)
            return loss

        h = forward_target()
        z = forward_context()
        loss = loss_fn(z, h)
        self.log("train/contrastive",
                 loss.detach().item(),
                 on_step=True,
                 on_epoch=True,
                 batch_size=imgs.shape[0],
                 rank_zero_only=True)
        self.train_loss.update(loss.detach().item(), weight=imgs.shape[0])
        return loss

    @torch.inference_mode()
    def validation_step(self, batch, _):
        imgs, masks_enc, masks_pred = batch
        imgs = imgs['image'].squeeze()

        def forward_target():
            with torch.no_grad():
                h = self.model.target_encoder(imgs)
                h = F.layer_norm(h, (h.size(-1), ))
                B = len(h)
                # -- create targets (masked regions of h)
                h = apply_masks(h, masks_pred)
                h = repeat_interleave_batch(h, B, repeat=len(masks_enc))
                return h

        def forward_context():
            z = self.model.encoder(imgs, masks_enc)
            z = self.model.predictor(z, masks_enc, masks_pred)
            return z

        def loss_fn(z, h):
            loss = self.criterion(z, h)
            return loss

        h = forward_target()
        z = forward_context()
        loss = loss_fn(z, h)
        self.val_loss.update(loss, weight=imgs.shape[0])
        return loss

    def on_train_epoch_end(self):
        train_loss = self.train_loss.compute()
        self.log("train/contrastive_manualepoch",
                 train_loss,
                 on_epoch=True,
                 sync_dist=True,
                 rank_zero_only=True)
        logging.info(f"train/contrastive_manualepoch {train_loss}")
        self.train_loss.reset()
        gc.collect()

    def on_validation_epoch_end(self):
        val_loss = self.val_loss.compute()
        self.log("val/contrastive_manualepoch",
                 val_loss,
                 on_epoch=True,
                 sync_dist=True,
                 rank_zero_only=True)
        logging.info(f"val/contrastive_manualepoch {val_loss}")
        self.val_loss.reset()
        gc.collect()

    def on_before_zero_grad(self, _):
        with torch.no_grad():
            for p, pt in zip(self.model.encoder.parameters(),
                             self.model.target_encoder.parameters()):
                pt.data = self.beta * pt.data + (1 - self.beta) * p.data

    def configure_optimizers(self):
        if not self.training_params_: return None  # Not training

        opt, sch = get_optimizer_scheduler(self.model,
                                           opt_cf=self.opt_cf_,
                                           schd_cf=self.schd_cf_,
                                           **self.training_params_)

        if sch:
            return [opt], {
                "scheduler": sch,
                "interval": "step",
                "frequency": 1,
                "name": "lr"
            }
        else:
            return [opt]


class InterPatchJEPASystem(EvalBaseSystem):
    def __init__(self,
                 model_hyperparams,
                 loss_params: Optional[Dict] = None,
                 ema_beta: Optional[float] = None,
                 opt_cf: Optional[Dict] = None,
                 schd_cf: Optional[Dict] = None,
                 training_params: Optional[Dict] = None):
        super().__init__()

        self.model = InterPatchJEPANetwork(**model_hyperparams)

        self.beta = ema_beta
        self.opt_cf_ = opt_cf
        self.schd_cf_ = schd_cf
        self.training_params_ = training_params

        if training_params:
            self.criterion = torch.nn.SmoothL1Loss(**loss_params)
        else:
            self.criterion = None

        self.train_loss = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()
        self.test_output = []

    def training_step(self, batch, _):
        target_hat, target_emb = self.model(batch)

        target_hat_gather = self.all_gather(target_hat, sync_grads=True)
        target_hat_gather = target_hat_gather.reshape(
            -1, *target_hat_gather.shape[-2:])

        target_emb_gather = self.all_gather(target_emb, sync_grads=True)
        target_emb_gather = target_emb_gather.reshape(
            -1, *target_emb_gather.shape[-2:])

        loss = self.criterion(target_hat_gather, target_emb_gather)
        self.log("train/contrastive",
                 loss.detach().item(),
                 on_step=True,
                 on_epoch=True,
                 batch_size=target_hat_gather.shape[0],
                 rank_zero_only=True)
        self.train_loss.update(loss.detach().item(),
                               weight=target_hat_gather.shape[0])

        return loss

    @torch.inference_mode()
    def validation_step(self, batch, _):

        target_hat, target_emb = self.model(batch)

        target_hat_gather = self.all_gather(target_hat, sync_grads=False)
        target_hat_gather = target_hat_gather.reshape(
            -1, *target_hat_gather.shape[-2:])

        target_emb_gather = self.all_gather(target_emb, sync_grads=False)
        target_emb_gather = target_emb_gather.reshape(
            -1, *target_emb_gather.shape[-2:])

        loss = self.criterion(target_hat_gather,
                              target_emb_gather).detach().item()
        self.val_loss.update(loss, weight=target_hat_gather.shape[0])

    def on_train_epoch_end(self):
        train_loss = self.train_loss.compute()
        self.log("train/contrastive_manualepoch",
                 train_loss,
                 on_epoch=True,
                 sync_dist=True,
                 rank_zero_only=True)
        logging.info(f"train/contrastive_manualepoch {train_loss}")
        self.train_loss.reset()
        gc.collect()

    def on_validation_epoch_end(self):
        val_loss = self.val_loss.compute()
        self.log("val/contrastive_manualepoch",
                 val_loss,
                 on_epoch=True,
                 sync_dist=True,
                 rank_zero_only=True)
        logging.info(f"val/contrastive_manualepoch {val_loss}")
        self.val_loss.reset()
        gc.collect()

    def on_before_zero_grad(self, _):
        with torch.no_grad():
            for p, pt in zip(self.model.bb.parameters(),
                             self.model.target_bb.parameters()):
                pt.data = self.beta * pt.data + (1 - self.beta) * p.data

    def configure_optimizers(self):
        if not self.training_params_: return None  # Not training

        opt, sch = get_optimizer_scheduler(self.model,
                                           opt_cf=self.opt_cf_,
                                           schd_cf=self.schd_cf_,
                                           **self.training_params_)

        if sch:
            return [opt], {
                "scheduler": sch,
                "interval": "step",
                "frequency": 1,
                "name": "lr"
            }
        else:
            return [opt]


#class VICRegSystemWithMask(VICRegSystem):
#
#    def __init__(self, loss_params, **kwargs):
#        super().__init__(**kwargs)
#
#        self.model = VICRegNetwork(**model_hyperparams)
#        if self.opt_cf_:
#            self.criterion = GeneralVICRegLoss(
#                embedding_dim=self.model.bb.num_out, **loss_params)
#
#        self.train_loss = torch.nn.ModuleDict({
#            n: torchmetrics.MeanMetric()
#            for n in GeneralVICRegLoss.get_loss_names()
#        })
#
#        self.val_loss = torch.nn.ModuleDict({
#            n: torchmetrics.MeanMetric()
#            for n in GeneralVICRegLoss.get_loss_names()
#        })
#
#        self.num_it_per_ep_ = num_it_per_ep
#
#    @torch.inference_mode()
#    def predict_step(self, batch, _):
#        self.model.eval()
#        assert batch["image"].shape[1] == 1
#        emb = self.model.bb(batch["image"][:, 0, ...], batch["mask"][:, 0,
#                                                                     ...])
#        return {
#            "path": batch["path"],
#            "label": batch["label"],
#            "embeddings": emb
#        }
#
#    def training_step(self, batch, _):
#        # forward pass
#        emb = [
#            self.model(batch["image"][:, i, ...], batch["mask"][:, i, ...])
#            for i in range(batch["image"].shape[1])
#        ]
#        emb = torch.stack(emb, dim=1)
#
#        emb_all = self.all_gather(emb, sync_grads=True)
#        emb_all = emb_all.reshape(-1, *emb_all.shape[-2:])
#
#        losses = self.criterion(emb_all)
#
#        bs = emb_all.shape[0] * torch.distributed.get_world_size()
#        for k in self.train_loss:
#            self.log(f"train/{k}",
#                     losses[k],
#                     on_step=True,
#                     on_epoch=False,
#                     batch_size=bs,
#                     rank_zero_only=True)
#            self.train_loss[k].update(losses[k], weight=bs)
#
#        return losses["loss"]
#
#    @torch.inference_mode()
#    def validation_step(self, batch, _):
#        emb = [
#            self.model(batch["image"][:, i, ...], batch["mask"][:, i, ...])
#            for i in range(batch["image"].shape[1])
#        ]
#        emb = torch.stack(emb, dim=1)
#
#        emb_all = self.all_gather(emb, sync_grads=True)
#        emb_all = emb_all.reshape(-1, *emb_all.shape[-2:])
#
#        losses = self.criterion(emb_all)
#
#        bs = emb_all.shape[0] * torch.distributed.get_world_size()
#        for k in self.val_loss.keys():
#            self.val_loss[k].update(losses[k], weight=bs)

#class SimSiamSystem(ContrastiveBaseSystem):
#    """SimSiam in Pytorch lightning implementation"""
#
#    def __init__(self, cf, num_it_per_ep):
#        super().__init__(cf, num_it_per_ep)
#        self.criterion = nn.CosineSimilarity(dim=1)
#        self.model = SimSiamNetwork(get_backbone(cf))
#
#    def forward(self, batch):
#        out1 = self.model(batch["image"][:, 0, ...])
#        out2 = self.model(batch["image"][:, 1, ...])
#
#        loss = -(self.criterion(out1['pred'], out2['proj'].detach()).mean(
#        ) + self.criterion(out2['pred'], out1['proj'].detach()).mean()) * 0.5
#        return loss
#
#    def training_step(self, batch, _):
#        loss = self.forward(batch)
#        bs = batch["image"].shape[0]
#        self.log("train/contrastive",
#                 loss,
#                 on_step=True,
#                 on_epoch=True,
#                 batch_size=bs,
#                 rank_zero_only=True)
#        self.train_loss.update(loss, weight=bs)
#        return loss
#
#    @torch.inference_mode()
#    def validation_step(self, batch, batch_idx):
#        bs = batch["image"].shape[0]
#        loss = self.forward(batch)
#        self.val_loss.update(loss, weight=bs)
#
#
#class BYOLSystem(ContrastiveBaseSystem):
#    """BYOL in Pytorch lightning implementation"""
#
#    def __init__(self, cf, num_it_per_ep):
#        super().__init__(cf, num_it_per_ep)
#        self.beta = cf["training"]["objective"]["params"]["ema_beta"]
#        # temporarily use simsiam mlp, will relax to general one in the future
#        self.model = SimSiamNetwork(get_backbone(cf))
#        self.target_model = self._get_target_model()
#
#    def forward(self, batch):
#        online_out1 = self.model(batch["image"][:, 0, ...])
#        online_out2 = self.model(batch["image"][:, 1, ...])
#
#        with torch.no_grad():
#            target_out1 = self.target_model(batch["image"][:, 0, ...])
#            target_out2 = self.target_model(batch["image"][:, 1, ...])
#            target_out1["proj"].detach_()
#            target_out2["proj"].detach_()
#
#        loss1 = self.criterion(online_out1["pred"],
#                               target_out2["proj"].detach())
#        loss2 = self.criterion(online_out2["pred"],
#                               target_out1["proj"].detach())
#        loss = loss1 + loss2
#        return loss.mean()
#
#    def training_step(self, batch, _):
#
#        loss = self.forward(batch)
#        bs = batch["image"].shape[0]
#        self.log("train/contrastive",
#                 loss,
#                 on_step=True,
#                 on_epoch=True,
#                 batch_size=bs,
#                 rank_zero_only=True)
#        self.train_loss.update(loss, weight=bs)
#        return loss
#
#    @torch.inference_mode()
#    def validation_step(self, batch, batch_idx):
#        loss = self.forward(batch)
#        bs = batch["image"].shape[0]
#        self.val_loss.update(loss, weight=bs)
#
#    def on_before_zero_grad(self, _):
#        self.update_target_encoder()
#
#    def criterion(self, x, y):
#        x = nn.functional.normalize(x, dim=-1, p=2)
#        y = nn.functional.normalize(y, dim=-1, p=2)
#        return 2 - 2 * (x * y).sum(dim=-1)
#
#    def _get_target_model(self):
#        target_model = copy.deepcopy(self.model)
#        for p in target_model.parameters():
#            p.requires_grad = False
#        return target_model
#
#    def update_target_encoder(self):
#        for p, pt in zip(self.model.parameters(),
#                         self.target_model.parameters()):
#            pt.data = self.beta * pt.data + (1 - self.beta) * p.data
#
