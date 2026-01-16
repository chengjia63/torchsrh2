import torch
import pytorch_lightning as pl
from dinov2_spy.models import build_model as build_spy_model
from dinov2.models import build_model as build_model
import dinov2.utils.utils as dinov2_utils


import uuid


class Dinov2EvalSystem(pl.LightningModule):

    def __init__(self, model_hyperparams, pretrained_weights,
                 get_image_attn=False, get_patch_tokens=False,
                 ckpt_key="teacher"):
        super().__init__()

        if get_image_attn:
            bm_func = build_spy_model
        else:
            bm_func = build_model

        self.get_image_attn = get_image_attn
        self.get_patch_tokens = get_patch_tokens
        self.teacher_backbone, _ = bm_func(model_hyperparams,
                                               only_teacher=True,
                                               img_size=model_hyperparams.get(
                                                   "img_size", 224))
        dinov2_utils.load_pretrained_weights(self.teacher_backbone,
                                             pretrained_weights, ckpt_key)
        self.teacher_backbone.eval()

    @torch.inference_mode()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        #if batch_idx==5:
        #    torch.save(batch, f"{uuid.uuid4().hex[:8]}.pt")
        
        if batch["image"].shape[1] == 1:
            if self.get_image_attn:
                emb = self.teacher_backbone(batch["image"][:, 0, ...], return_attn=True)
                results = {
                    "path": [i for i in batch["path"]],
                    "label": batch["label"],
                    "embeddings": emb["x_norm_clstoken"],
                    "attns": emb["attns"]
                }

                if self.get_patch_tokens:
                    results.update({
                        "patch_embeddings": emb["x_norm_patchtokens"]
                    })
            
            elif self.get_patch_tokens:
                emb, full_dict = self.teacher_backbone(batch["image"][:, 0, ...])
                results = {
                    "path": batch["path"],
                    "label": batch["label"],
                    "embeddings": emb,
                    "patch_embeddings": full_dict["x_norm_patchtokens"],
                }

            else:
                emb, _ = self.teacher_backbone(batch["image"][:, 0, ...])
                results = {
                    "path": batch["path"],
                    "label": batch["label"],
                    "embeddings": emb,
                }
            return results
        else:
            assert not self.get_image_attn
            assert not self.get_patch_tokens

            embs = [self.teacher_backbone(batch["image"][:, i, ...])[0]
                    for i in range(batch["image"].shape[1])]

            return {
                "path": batch["path"],
                "label": batch["label"],
                "embeddings": torch.stack(embs, dim=1),
            }
