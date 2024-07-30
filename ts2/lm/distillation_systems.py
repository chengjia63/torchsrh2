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
from ts2.models.ssl import (instantiate_backbone, MLP,
                            ContrastiveLearningNetwork, VICRegNetwork)
#SimSiamNetwork)  #, VICRegNetworkWithMask)
from ts2.models.pjepa import InterPatchJEPANetwork
from ts2.models.ijepa import IJEPANetwork, apply_masks, repeat_interleave_batch
from ts2.optim.utils import get_optimizer_scheduler
from ts2.lm.ssl_systems import EvalBaseSystem


class CommitteeDistillationNetwork(torch.nn.Module):
    """A network consists of a backbone and projection head.
    Forward pass returns the normalized embeddings after a projection layer.
    """

    def __init__(self,
                 backbone_cf: Dict,
                 student_proj_params: Dict = None,
                 pred_params: Dict = None):
        super(CommitteeDistillationNetwork, self).__init__()
        self.bb = instantiate_backbone(**backbone_cf)

        if student_proj_params:
            self.student_projector = MLP(n_in=self.bb.num_out,
                                         **student_proj_params)
        else:
            self.student_projector = torch.nn.Identity()
        #self.teacher_projectors = torch.nn.ModuleList(
        #    [MLP(**projp) for projp in teacher_proj_params])

        if pred_params:
            self.predictors = torch.nn.ModuleList(
                [MLP(**projp) for projp in pred_params])

        self.num_out = None

    def forward(self, batch: Dict) -> torch.Tensor:
        emb = [
            self.bb(batch["image"][i, ...])
            for i in range(batch["image"].shape[0])
        ]
        emb = self.student_projector(torch.cat(emb, dim=0))
        pred = [F.normalize(pd(emb), p=2.0, dim=1) for pd in self.predictors]
        with torch.no_grad():
            target = [
                F.normalize(fm_emb.view(-1, fm_emb.shape[-1]), p=2.0, dim=1)
                for fm_emb in batch["fm_embs"]
            ]
        return pred, target


class CommitteeDistillationBaseSystem(EvalBaseSystem):
    """Lightning system for SimCLR experiment"""

    def __init__(self, opt_cf, schd_cf, training_params):
        super().__init__()

        self.opt_cf_ = opt_cf
        self.schd_cf_ = schd_cf
        self.training_params_ = training_params

        self.train_loss = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()

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

    @torch.inference_mode()
    def on_validation_epoch_end(self):
        val_loss = self.val_loss.compute()
        self.log("val/loss_manualepoch",
                 val_loss,
                 on_epoch=True,
                 sync_dist=True,
                 rank_zero_only=True)
        logging.info(f"val/loss_manualepoch {val_loss}")
        self.val_loss.reset()

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


class CommitteeDistillationSystem(CommitteeDistillationBaseSystem):
    """Lightning system for SimCLR experiment"""

    def __init__(self,
                 model_hyperparams,
                 loss_params: Optional[Dict] = None,
                 opt_cf: Optional[Dict] = None,
                 schd_cf: Optional[Dict] = None,
                 training_params: Optional[Dict] = None):
        super().__init__(opt_cf=opt_cf,
                         schd_cf=schd_cf,
                         training_params=training_params)

        self.model = CommitteeDistillationNetwork(**model_hyperparams)

        if self.training_params_:
            self.criterion = torch.nn.SmoothL1Loss(**loss_params)
            #self.criterion = torch.nn.L1Loss(**loss_params)
        else:
            self.criterion = None

    def forward(self, data):
        return torch.cat([self.model(x)["emb"] for x in data["image"]], dim=1)

    def training_step(self, batch, _):
        pred, target = self.model(batch)

        losses = [self.criterion(th, te) for th, te in zip(pred, target)]

        for i, ell in enumerate(losses):
            self.log(f"train/contrastive_{i}",
                     ell.detach().item(),
                     on_step=True,
                     on_epoch=True,
                     batch_size=pred[0].shape[0],
                     rank_zero_only=True)

        all_loss = torch.stack(losses).mean()
        self.log("train/contrastive",
                 all_loss.detach().item(),
                 on_step=True,
                 on_epoch=True,
                 batch_size=pred[0].shape[0],
                 rank_zero_only=True)
        self.train_loss.update(all_loss.detach().item(),
                               weight=pred[0].shape[0])

        return all_loss

    @torch.inference_mode()
    def validation_step(self, batch, _):
        pred, target = self.model(batch)

        all_loss = torch.stack(
            [self.criterion(th, te) for th, te in zip(pred, target)]).mean()

        self.val_loss.update(all_loss, weight=pred[0].shape[0])


class CRDNetwork(torch.nn.Module):
    """A network consists of a backbone and projection head.
    Forward pass returns the normalized embeddings after a projection layer.
    """

    def __init__(self,
                 backbone_cf: Dict,
                 student_proj_params: Dict = None,
                 teacher_proj_params: Dict = None,
                 pred_params: Dict = None):
        super(CRDNetwork, self).__init__()
        self.bb = instantiate_backbone(**backbone_cf)

        if student_proj_params:
            self.student_projector = MLP(n_in=self.bb.num_out,
                                         **student_proj_params)
        else:
            self.student_projector = torch.nn.Identity()

        if teacher_proj_params:
            self.teacher_projectors = torch.nn.ModuleList(
                [MLP(**projp) for projp in teacher_proj_params])
        else:
            self.teacher_projectors = torch.nn.Identity()

        if pred_params:
            self.predictors = torch.nn.ModuleList(
                [MLP(**projp) for projp in pred_params])
        else:
            self.predictors = torch.nn.Identity()

        self.num_out = None

    def forward(self, batch: Dict) -> torch.Tensor:
        emb = [
            self.student_projector(self.bb(batch["image"][:, i, ...]))
            for i in range(batch["image"].shape[1])
        ]

        pred = [
            torch.stack(
                [F.normalize(pd(emb_view), p=2.0, dim=-1) for emb_view in emb],
                dim=1) for pd in self.predictors
        ]

        target = [
            F.normalize(tproj(fm_emb[:, 0, ...]), p=2.0,
                        dim=1).unsqueeze(dim=1)
            for tproj, fm_emb in zip(self.teacher_projectors, batch["fm_embs"])
        ]
        return pred, target


class CRDDistillationSystem(CommitteeDistillationBaseSystem):

    def __init__(self,
                 model_hyperparams,
                 loss_params: Optional[Dict] = None,
                 opt_cf: Optional[Dict] = None,
                 schd_cf: Optional[Dict] = None,
                 training_params: Optional[Dict] = None):

        super().__init__(opt_cf=opt_cf,
                         schd_cf=schd_cf,
                         training_params=training_params)

        self.model = CRDNetwork(**model_hyperparams)
        if self.training_params_:
            self.criterion = torch.nn.ModuleList([
                SupConLoss(**loss_params)
                for _ in range(len(model_hyperparams.pred_params))
            ])
        else:
            self.criterion = None

    def training_step(self, batch, _):
        pred, target = self.model(batch)

        pred = [
            self.all_gather(p, sync_grads=True).reshape(-1, *p.shape[1:])
            for p in pred
        ]
        target = [
            self.all_gather(t, sync_grads=True).reshape(-1, *t.shape[1:])
            for t in target
        ]

        losses = [
            crit(torch.cat([th, te], dim=1))["loss"]
            for crit, th, te in zip(self.criterion, pred, target)
        ]

        for i, ell in enumerate(losses):
            self.log(f"train/contrastive_{i}",
                     ell.detach().item(),
                     on_step=True,
                     on_epoch=True,
                     batch_size=pred[0].shape[0],
                     rank_zero_only=True)

        all_loss = torch.stack(losses).mean()
        self.log("train/contrastive",
                 all_loss.detach().item(),
                 on_step=True,
                 on_epoch=True,
                 batch_size=pred[0].shape[0],
                 rank_zero_only=True)
        self.train_loss.update(all_loss.detach().item(),
                               weight=pred[0].shape[0])

        return all_loss

    @torch.inference_mode()
    def validation_step(self, batch, _):
        pred, target = self.model(batch)
        pred = [
            self.all_gather(p, sync_grads=True).reshape(-1, *p.shape[1:])
            for p in pred
        ]
        target = [
            self.all_gather(t, sync_grads=True).reshape(-1, *t.shape[1:])
            for t in target
        ]
        all_loss = torch.stack([
            crit(torch.cat([th, te], dim=1))["loss"]
            for crit, th, te in zip(self.criterion, pred, target)
        ]).mean()

        self.val_loss.update(all_loss, weight=pred[0].shape[0])
