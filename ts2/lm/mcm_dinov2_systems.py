import logging
import itertools
import copy
from functools import partial
from typing import Dict, Any, Optional
import gc
import torch
from torch import nn
import torch.nn as nn
import torch.nn.functional as F
import einops

import pytorch_lightning as pl
import torchmetrics

import random
import math
import numpy as np
#from torchsrh.losses.supcon import SupConLoss
#from torchsrh.losses.vicreg import GeneralVICRegLoss
#from ts2.models.ssl import (instantiate_backbone, MLP)
from ts2.lm.ssl_systems import EvalBaseSystem
from ts2.lm.mcm_systems import sample_view, crop_views
from torchvision import transforms as transforms
from ts2.optim.utils import get_optimizer_scheduler  

from ts2.models.dinov2.ssl_meta_arch import SSLMetaArch


class CosineScheduler(object):

    def __init__(self,
                 base_value,
                 final_value,
                 total_iters,
                 warmup_iters=0,
                 start_warmup_value=0,
                 freeze_iters=0):
        super().__init__()
        self.final_value = final_value
        self.total_iters = total_iters

        freeze_schedule = np.zeros((freeze_iters))

        warmup_schedule = np.linspace(start_warmup_value, base_value,
                                      warmup_iters)

        iters = np.arange(total_iters - warmup_iters - freeze_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (
            1 + np.cos(np.pi * iters / len(iters)))
        self.schedule = np.concatenate(
            (freeze_schedule, warmup_schedule, schedule))

        assert len(self.schedule) == self.total_iters

    def __getitem__(self, it):
        if it >= self.total_iters:
            return self.final_value
        else:
            return self.schedule[it]


def apply_scaling_rule(base_lr, effective_bs, scaling_rule):  # to fix
    if scaling_rule == "sqrt_wrt_1024":
        lr = base_lr * math.sqrt(effective_bs / 1024.0)
        logging.info(f"sqrt scaling learning rate; base: {base_lr}, new: {lr}")
        return lr
    else:
        raise NotImplementedError


class MCMDinov2System(EvalBaseSystem):

    def __init__(self,
                 model_hyperparams,
                 #opt_cf: Optional[Dict] = None,
                 #schd_cf: Optional[Dict] = None,
                 training_params: Optional[Dict] = None):
        super().__init__()

        #self.opt_cf_ = opt_cf
        #self.schd_cf_ = schd_cf

        self.training_params_ = training_params
        print(training_params)

        self.tile_encoder = SSLMetaArch(model_hyperparams.tile_encoder)
        #self.patch_encoder = SSLMetaArch(model_hyperparams.patch_encoder)

        if training_params:
            model_hyperparams.optim["lr"] = apply_scaling_rule(
                model_hyperparams.optim.base_lr,
                training_params["effective_batch_size"],
                model_hyperparams.optim.scaling_rule)
            self.model_hyperparams_ = model_hyperparams
            self.loss_coeff = model_hyperparams.loss_coeff

            self.train_loss = torchmetrics.MeanMetric()
            self.val_loss = torchmetrics.MeanMetric()

            self.patch_cropping_params = model_hyperparams.patch_cropping

            self.build_schedulers(model_hyperparams,
                                  training_params["num_it_per_ep"],
                                  training_params["num_ep_total"])

        else:
            self.criterion = None

    def build_schedulers(self, cfg, OFFICIAL_EPOCH_LENGTH, num_ep_total):
        lr = dict(
            base_value=cfg.optim["lr"],
            final_value=cfg.optim["min_lr"],
            total_iters=num_ep_total * OFFICIAL_EPOCH_LENGTH,
            warmup_iters=cfg.optim["warmup_epochs"] * OFFICIAL_EPOCH_LENGTH,
            start_warmup_value=0,
        )
        wd = dict(
            base_value=cfg.optim["weight_decay"],
            final_value=cfg.optim["weight_decay_end"],
            total_iters=num_ep_total * OFFICIAL_EPOCH_LENGTH,
        )
        momentum = dict(
            base_value=cfg.tile_encoder.teacher["momentum_teacher"],
            final_value=cfg.tile_encoder.teacher["final_momentum_teacher"],
            total_iters=num_ep_total * OFFICIAL_EPOCH_LENGTH,
        )
        tile_teacher_temp = dict(
            base_value=cfg.tile_encoder.teacher["teacher_temp"],
            final_value=cfg.tile_encoder.teacher["teacher_temp"],
            total_iters=cfg.tile_encoder.teacher["warmup_teacher_temp_epochs"]
           * OFFICIAL_EPOCH_LENGTH,
            warmup_iters=cfg.tile_encoder.teacher["warmup_teacher_temp_epochs"]
            * OFFICIAL_EPOCH_LENGTH,
            start_warmup_value=cfg.tile_encoder.teacher["warmup_teacher_temp"],
        )

        self.lr_schedule = CosineScheduler(**lr)
        self.wd_schedule = CosineScheduler(**wd)
        self.momentum_schedule = CosineScheduler(**momentum)
        self.tile_teacher_temp_schedule = CosineScheduler(**tile_teacher_temp)
        self.last_layer_lr_schedule = CosineScheduler(**lr)

        self.last_layer_lr_schedule.schedule[:cfg.optim[
            "freeze_last_layer_epochs"] * OFFICIAL_EPOCH_LENGTH] = 0  # mimicking the original schedules

        logging.info("Schedulers ready.")

    def training_step(self, batch, _):
        # 48x48 tile encoding
        #torch.save(batch, "dinov2_2.pt")
        tile_loss, tile_loss_comp = self.tile_encoder(
            batch,
            teacher_temp=self.tile_teacher_temp_schedule[
                self.trainer.global_step])

        #import pdb
        #pdb.set_trace()
        for l in tile_loss_comp:
            self.log(f"train/tile_{l}",
                     tile_loss_comp[l].detach().item(),
                     on_step=True,
                     on_epoch=True,
                     batch_size=batch["collated_global_crops"].shape[0],
                     rank_zero_only=True)

        self.log(f"train/tile_loss",
                 tile_loss.detach().item(),
                 on_step=True,
                 on_epoch=True,
                 batch_size=batch["collated_global_crops"].shape[0],
                 rank_zero_only=True)
        self.train_loss.update(tile_loss.detach().item(),
                               weight=batch["collated_global_crops"].shape[0])

        self.log(f"hyperparams/lr",
                 self.lr_schedule[self.trainer.global_step],
                 on_step=True,
                 rank_zero_only=True)
        self.log(f"hyperparams/wd",
                 self.wd_schedule[self.trainer.global_step],
                 on_step=True,
                 rank_zero_only=True)
        self.log(f"hyperparams/mom",
                 self.momentum_schedule[self.trainer.global_step],
                 on_step=True,
                 rank_zero_only=True)
        self.log(f"hyperparams/tile_teacher_temp_schedule",
                 self.tile_teacher_temp_schedule[self.trainer.global_step],
                 on_step=True,
                 rank_zero_only=True)
        self.log(f"hyperparams/last_layer_lr_schedule",
                 self.last_layer_lr_schedule[self.trainer.global_step],
                 on_step=True,
                 rank_zero_only=True)

        return tile_loss

        # ================================

        #student_token_global_reshaped = einops.rearrange(
        #    student_global_bb_emb[:, 0, :],
        #    "(v b nh nw) d -> v b d nh nw",
        #    v=2,
        #    nh=6,
        #    nw=6)
        #student_token_local_reshaped = einops.rearrange(
        #    student_local_bb_emb[:, 0, :],
        #    "(v b nh nw) d -> v b d nh nw",
        #    v=len(region_crops) - 2,
        #    nh=6,
        #    nw=6)
        #student_tokens = torch.cat(
        #    (student_token_global_reshaped, student_token_local_reshaped))

        #with torch.no_grad():
        #    teacher_tokens = einops.rearrange(teacher_bb_emb[:, 0, :],
        #                                      "(v b nh nw) d -> v b d nh nw",
        #                                      v=2,
        #                                      nh=6,
        #                                      nw=6)

        #sampled_student_tokens = sample_view(
        #    student_tokens, self.patch_cropping_params.sample_min,
        #    self.patch_cropping_params.sample_max)
        #with torch.no_grad():
        #    sampled_teacher_tokens = sample_view(
        #        teacher_tokens, self.patch_cropping_params.sample_min,
        #        self.patch_cropping_params.sample_max)

        #cropped_global_tokens = [
        #    crop_views([sampled_student_tokens, sampled_teacher_tokens],
        #               h0=self.patch_cropping_params.global_size,
        #               w0=self.patch_cropping_params.global_size)
        #    for _ in range(self.patch_cropping_params.global_crops_number)
        #]
        #student_global_tokens = [i[0] for i in cropped_global_tokens]
        #teacher_global_tokens = [i[1] for i in cropped_global_tokens]
        #student_local_tokens = [
        #    crop_views([sampled_student_tokens],
        #               h0=self.patch_cropping_params.local_size,
        #               w0=self.patch_cropping_params.local_size)[0]
        #    for _ in range(self.patch_cropping_params.local_crops_number)
        #]

        #patch_masks = [
        #    self.patch_mask_generator.get_mask(i).to(i.device)
        #    for i in student_global_tokens
        #]

        #student_patch_global_emb = self.student_patch_encoder(
        #    student_global_tokens, mask=patch_masks)

        #with torch.no_grad():
        #    teacher_patch_emb = self.teacher_patch_encoder(
        #        teacher_global_tokens)

        #self.student_patch_encoder.backbone.masked_im_modeling = False
        #student_patch_local_emb = self.student_patch_encoder(
        #    student_local_tokens)
        #self.student_patch_encoder.backbone.masked_im_modeling = True

        #torch.save(
        #    {
        #        "t": teacher_tokens,
        #        "ts": sampled_teacher_tokens,
        #        "s": student_tokens,
        #        "ss": sampled_student_tokens
        #    }, "stage1_out.pt")

        #all_loss = self.criterion(student_global_emb, teacher_emb,
        #                          student_local_emb[0], tile_masks,
        #                          self.current_epoch)
        ##all_loss = {}
        #patch_loss = self.patch_criterion(student_patch_global_emb,
        #                                  teacher_patch_emb,
        #                                  student_patch_local_emb[0],
        #                                  patch_masks, self.current_epoch)

        #for k in patch_loss:
        #    all_loss[f"patch_{k}"] = patch_loss[k]

        #for l in all_loss:
        #    self.log(f"train/{l}",
        #             all_loss[l].detach().item(),
        #             on_step=True,
        #             on_epoch=True,
        #             batch_size=batched_regions.shape[0],
        #             rank_zero_only=True)

        #loss = (all_loss.pop("loss") * self.loss_coeff.tile +
        #        all_loss.pop("patch_loss") * self.loss_coeff.patch)
        ##loss = all_loss.pop("patch_loss")

        #self.train_loss.update(loss.detach().item(),
        #                       weight=batched_regions.shape[0])

        #self.log(f"train/combined_loss",
        #         loss.detach().item(),
        #         on_step=True,
        #         on_epoch=True,
        #         batch_size=batched_regions.shape[0],
        #         rank_zero_only=True)
        #return loss

    @torch.inference_mode()
    def validation_step(self, batch, _):
        return

    @torch.inference_mode()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        raise NotImplementedError()
        emb = self.teacher_tile_encoder.backbone(batch["image"][:, 0,
                                                                ...])[:, 0, :]
        results = {
            "path": batch["path"],
            "label": batch["label"],
            "embeddings": emb
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
        return

    def on_before_zero_grad(self, _):
        with torch.no_grad():
            mom = self.momentum_schedule[self.trainer.global_step]
            self.tile_encoder.update_teacher(mom)

    def on_before_optimizer_step(self, optimizer):
        wd = self.wd_schedule[self.trainer.global_step]
        lr = self.lr_schedule[self.trainer.global_step]
        last_layer_lr = self.last_layer_lr_schedule[self.trainer.global_step]

        for param_group in optimizer.param_groups:
            is_last_layer = param_group["is_last_layer"]
            lr_multiplier = param_group["lr_multiplier"]
            wd_multiplier = param_group["wd_multiplier"]
            param_group["weight_decay"] = wd * wd_multiplier
            param_group["lr"] = (last_layer_lr
                                 if is_last_layer else lr) * lr_multiplier

    def configure_optimizers(self):
        if not self.training_params_: return None  # Not training
        #import pdb; pdb.set_trace()
        return torch.optim.AdamW(
            self.tile_encoder.get_params_groups(),
            betas=(self.model_hyperparams_.optim.adamw_beta1,
                   self.model_hyperparams_.optim.adamw_beta2))

    #def configure_optimizers(self):
    #    if not self.training_params_: return None  # Not training

    #    opt, sch = get_optimizer_scheduler(self.tile_encoder.student,
    #                                       opt_cf=self.opt_cf_,
    #                                       schd_cf=self.schd_cf_,
    #                                       **self.training_params_)

    #    if sch:
    #        return [opt], {
    #            "scheduler": sch,
    #            "interval": "step",
    #            "frequency": 1,
    #            "name": "lr"
    #        }
    #    else:
    #        return opt


class CellIBOTSystem(EvalBaseSystem):

    def __init__(self,
                 model_hyperparams,
                 opt_cf: Optional[Dict] = None,
                 schd_cf: Optional[Dict] = None,
                 training_params: Optional[Dict] = None):
        super().__init__()

        self.opt_cf_ = opt_cf
        self.schd_cf_ = schd_cf

        self.training_params_ = training_params
        print(training_params)

        if training_params:
            # tile level encoder
            self.student_tile_encoder = vit_small(
                **model_hyperparams.student_tile_encoder)

            self.teacher_tile_encoder = vit_small(
                **model_hyperparams.teacher_tile_encoder)

            self.student_tile_encoder = MultiCropWrapper(
                self.student_tile_encoder,
                iBOTHead(**model_hyperparams.student_tile_ibot_head))

            self.teacher_tile_encoder = MultiCropWrapper(
                self.teacher_tile_encoder,
                iBOTHead(**model_hyperparams.teacher_tile_ibot_head))

            # teacher and student start with the same weights
            self.teacher_tile_encoder.load_state_dict(
                self.student_tile_encoder.state_dict(), strict=False)
            # there is no backpropagation through the teacher, so no need for gradients
            for p in self.teacher_tile_encoder.parameters():
                p.requires_grad = False
        else:
            self.teacher_tile_encoder = vit_small(
                **model_hyperparams.teacher_tile_encoder)

            self.teacher_tile_encoder = MultiCropWrapper(
                self.teacher_tile_encoder)

        if training_params:
            self.criterion = iBOTLoss(
                **model_hyperparams.tile_ibot_loss,
                warmup_teacher_temp_epochs=int(
                    training_params["num_ep_total"] *
                    0.3),  #args.warmup_teacher_temp_epochs
                nepochs=training_params["num_ep_total"]  # TODO #args.epochs
            )

            self.train_loss = torchmetrics.MeanMetric()
            self.val_loss = torchmetrics.MeanMetric()
            #self.test_output = []

            self.tile_xform = DataAugmentationiBOT(
                **model_hyperparams.tile_xform)
            self.tile_mask_generator = IBotMaskGenerator(
                **model_hyperparams.tile_mask_generator)

            self.momentum_scheduler = cosine_scheduler(
                epochs=training_params["num_ep_total"],
                niter_per_ep=self.training_params_["num_it_per_ep"],
                **model_hyperparams.momentum_scheduler)
        else:
            self.criterion = None

    def training_step(self, batch, _):

        with torch.no_grad():
            self.tile_mask_generator.set_epoch(self.current_epoch)

            region_crops = self.tile_xform(batch["image"].squeeze(dim=1))

            tile_masks = [
                self.tile_mask_generator.get_mask(i).to(batch["image"].device)
                for i in region_crops[:2]
            ]

        #torch.save({"region_crops": region_crops, "tile_masks": tile_masks}, "test_aug_smpt.pt")

        student_global_bb_emb, student_global_emb = self.student_tile_encoder(
            region_crops[:2], mask=tile_masks, return_backbone_feat=True)

        with torch.no_grad():
            teacher_bb_emb, teacher_emb = self.teacher_tile_encoder(
                region_crops[:2], return_backbone_feat=True)

        self.student_tile_encoder.backbone.masked_im_modeling = False
        student_local_bb_emb, student_local_emb = self.student_tile_encoder(
            region_crops[2:], return_backbone_feat=True)
        self.student_tile_encoder.backbone.masked_im_modeling = True

        all_loss = self.criterion(student_global_emb, teacher_emb,
                                  student_local_emb[0], tile_masks,
                                  self.current_epoch)

        for l in all_loss:
            self.log(f"train/{l}",
                     all_loss[l].detach().item(),
                     on_step=True,
                     on_epoch=True,
                     batch_size=batch["image"].shape[0],
                     rank_zero_only=True)

        loss = all_loss.pop("loss")

        self.train_loss.update(loss.detach().item(),
                               weight=batch["image"].shape[0])

        return loss

    @torch.inference_mode()
    def validation_step(self, batch, _):
        return

    @torch.inference_mode()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        emb = self.teacher_tile_encoder.backbone(batch["image"][:, 0,
                                                                ...])[:, 0, :]
        results = {
            "path": batch["path"],
            "label": batch["label"],
            "embeddings": emb
        }

        return results

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
        return

    def on_before_zero_grad(self, _):
        names_q, params_q, names_k, params_k = [], [], [], []
        for name_q, param_q in self.student_tile_encoder.named_parameters():
            names_q.append(name_q)
            params_q.append(param_q)
        for name_k, param_k in self.teacher_tile_encoder.named_parameters():
            names_k.append(name_k)
            params_k.append(param_k)
        names_common = list(set(names_q) & set(names_k))
        params_q = [
            param_q for name_q, param_q in zip(names_q, params_q)
            if name_q in names_common
        ]
        params_k = [
            param_k for name_k, param_k in zip(names_k, params_k)
            if name_k in names_common
        ]

        with torch.no_grad():
            m = self.momentum_scheduler[self.trainer.global_step]
            for param_q, param_k in zip(params_q, params_k):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

    def configure_optimizers(self):
        if not self.training_params_: return None  # Not training

        opt, sch = get_optimizer_scheduler(self,
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
            return opt
