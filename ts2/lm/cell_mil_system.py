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
from ts2.optim.utils import get_optimizer_scheduler
from ts2.models.ibot_vit import vit_small   

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.M = 500
        self.L = 128
        self.ATTENTION_BRANCHES = 1

        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh(),
            nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.M*self.ATTENTION_BRANCHES, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        H = x #self.feature_extractor_part2(H)  # KxM

        A = self.attention(H)  # KxATTENTION_BRANCHES
        A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxK
        A = F.softmax(A, dim=1)  # softmax over K

        Z = torch.mm(A, H)  # ATTENTION_BRANCHESxM

        Y_prob = self.classifier(Z)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A

class GatedAttention(nn.Module):
    def __init__(self):
        super(GatedAttention, self).__init__()
        self.M = 384
        self.L = 128
        self.ATTENTION_BRANCHES = 1
        self.nc = 2

        self.attention_V = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix U
            nn.Sigmoid()
        )

        self.attention_w = nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)

        self.classifier = nn.Sequential(
            nn.Linear(self.M*self.ATTENTION_BRANCHES, self.nc),
#            nn.Sigmoid()
        )

    def forward(self, x):

        H = x #self.feature_extractor_part2(H)  # KxM

        A_V = self.attention_V(H)  # KxL
        A_U = self.attention_U(H)  # KxL
        A = self.attention_w(A_V * A_U) # element wise multiplication # KxATTENTION_BRANCHES
        A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxK
        A = F.softmax(A, dim=1)  # softmax over K

        Z = torch.mm(A, H)  # ATTENTION_BRANCHESxM

        Y_prob = self.classifier(Z)

        return Y_prob, A, Z

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A
    
class CellABMILSystem(pl.LightningModule):
    """Lightning system for contrastive learning experiments."""

    def __init__(self,
                 model_hyperparams,
                 opt_cf: Optional[Dict] = None,
                 schd_cf: Optional[Dict] = None,
                 loss_params=None, 
                 training_params: Optional[Dict] = None):
        super().__init__()

        self.opt_cf_ = opt_cf
        self.schd_cf_ = schd_cf
        self.training_params_ = training_params

        self.tile_backbone = vit_small(**model_hyperparams.backbone)
        bbone_ckpt = torch.load("/nfs/turbo/umms-tocho-snr/exp/chengjia/ts2/scsrh_ibot/5b0f78d2_Mar17-01-39-03_sd1000_ibot_1000_lr0.001_tune0/models/ckpt-epoch64-step433875-loss0.00.ckpt", weights_only=False)
        self.tile_backbone.load_state_dict({k.removeprefix("teacher_tile_encoder.backbone."): bbone_ckpt["state_dict"][k] for k in bbone_ckpt["state_dict"].keys() if k.startswith("teacher_tile_encoder.backbone.")})

        self.model = GatedAttention()

        self.train_loss = torchmetrics.MeanMetric()
        self.val_acc = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()

        if self.opt_cf_:
            self.criterion = torch.nn.CrossEntropyLoss(**loss_params)

    def forward(self, data):
        return torch.cat([self.model(x)["emb"] for x in data["image"]], dim=1)



    def training_step(self, batch, _):
        bs = batch["pixels"][0].shape[0] * torch.distributed.get_world_size()

        with torch.inference_mode():
            feat = [self.tile_backbone(bp) for bp in batch["pixels"]]
            cls_tokens = [f[:,0,:] for f in feat]

        out = [self.model(ct) for ct in cls_tokens]
        logits = torch.cat([o[0] for o in out])
        y = batch["label"]

        loss = self.criterion(logits, y)
        self.log("val/loss",
                 loss.detach().item(),
                 batch_size=bs,
                 on_step=True,
                 on_epoch=True,
                 sync_dist=True,
                 rank_zero_only=True)
        self.train_loss.update(loss.detach().item(), weight=bs)
        return loss

    @torch.inference_mode()
    def validation_step(self, batch, batch_idx):
        bs = batch["pixels"][0].shape[0] * torch.distributed.get_world_size()

        feat = [self.tile_backbone(bp) for bp in batch["pixels"]]
        cls_tokens = [f[:,0,:] for f in feat]

        out = [self.model(ct) for ct in cls_tokens]
        logits = torch.cat([o[0] for o in out])
        y = batch["label"]

        loss = self.criterion(logits, y)
        acc = torchmetrics.functional.accuracy(torch.nn.functional.softmax(logits, dim=1)[:,1]>0.5, y, num_classes=2, average="macro")

        self.val_loss.update(loss, weight=bs)
        self.val_acc.update(acc, weight=bs)

        self.log("val/loss",
                 loss.detach().item(),
                 batch_size=bs,
                 on_epoch=True,
                 sync_dist=True,
                 rank_zero_only=True)
        self.log("val/acc",
                 loss.detach().item(),
                 on_epoch=True,
                 sync_dist=True,
                 batch_size=bs,
                 rank_zero_only=True)
    
    @torch.inference_mode()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):

        feat = [self.tile_backbone(bp) for bp in batch["pixels"]]
        cls_tokens = [f[:,0,:] for f in feat]

        out = [self.model(ct) for ct in cls_tokens]
        logits = torch.cat([o[0] for o in out])
        logits_scores = [o[1] for o in out]
        embs = [o[2] for o in out]

        results = {
            "cls": cls_tokens,
            "logits": logits,
            "attn": logits_scores,
            "embs": embs,
            "label": batch["label"],
            "path": batch["path"]
        }

        return results

    def on_train_epoch_end(self):
        train_loss = self.train_loss.compute()
        self.log("train/loss_manualepoch",
                 train_loss,
                 on_epoch=True,
                 sync_dist=True,
                 rank_zero_only=True)
        logging.info(f"train/loss_manualepoch {train_loss}")
        self.train_loss.reset()
        gc.collect()

    def on_validation_epoch_end(self):
        val_loss = self.val_loss.compute()
        val_acc = self.val_acc.compute()
        self.log("val/loss_manualepoch",
                 val_loss,
                 on_epoch=True,
                 sync_dist=True,
                 rank_zero_only=True)
        self.log("val/acc__manualepoch",
                 val_acc,
                 on_epoch=True,
                 sync_dist=True,
                 rank_zero_only=True)
        logging.info(f"val/loss_manualepoch {val_loss}")
        logging.info(f"val/acc_manualepoch {val_acc}")
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

