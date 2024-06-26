import os
from os.path import join as opj
from datetime import datetime
import uuid
import gzip
import logging
from tqdm import tqdm
from PIL import Image
import torch
import torchvision
import pytorch_lightning as pl
from omegaconf import OmegaConf
from typing import Dict, Any
import einops
from ts2.data.histology_data_module import PatchDataModule

from ts2.train.infra import (parse_args, read_process_cf, setup_infra_training,
                             setup_infra_testing, get_rank)
from transformers import AutoImageProcessor, AutoModel
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms
import timm

import collections
import torch.nn as nn

def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)


class ConvStem(nn.Module):

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()

        assert patch_size == 4
        assert embed_dim % 8 == 0

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten


        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for l in range(2):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

class CTransPathSystem(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model =  timm.create_model('swin_tiny_patch4_window7_224', embed_layer=ConvStem, pretrained=False)
        self.model.head = nn.Identity()
        td = torch.load("/nfs/mm-isilon/brainscans/dropbox/exp/renly/ckpts/ctranspath/ctranspath.pth")
        self.model.load_state_dict(td['model'], strict=True)

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        self.transform = transforms.Normalize(mean = mean, std = std)

    @torch.inference_mode()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        out = self.model.forward(self.transform(batch["image"].squeeze()))
        return {
            "path": batch["path"],
            "label": batch["label"],
            "embeddings": out
        }
        

#class ImageNetResnetSystem(pl.LightningModule):
#    def __init__(self):
#        super().__init__()
#
#    @torch.inference_mode()
#    def predict_step(self, batch, batch_idx, dataloader_idx=0):
#        pass
@torch.inference_mode()
def main():

    cf = read_process_cf(parse_args())
    dm = PatchDataModule(config=cf)

    #model = UNISystem()
    #model = PLIPSystem()
    #model = ConchSystem()
    #model = VirchowSystem()
    #model = GigapathSystem()
    #model = ImageNetResnetSystem()
    #model = DINOv2System()
    #model = CLIPSystem()
    #model = HIPTSystem()
    model = CTransPathSystem()

    pred_trainer = pl.Trainer(accelerator="gpu",
                              devices=1,
                              default_root_dir=".",
                              inference_mode=True)  # deterministic=True)

    pred_raw = pred_trainer.predict(model, datamodule=dm)

    def process_predictions(predictions):
        pred = {
            "embeddings": torch.cat([p["embeddings"] for p in predictions]),
            "label": torch.cat([p["label"] for p in predictions]),
            "path": [pk for p in predictions for pk in p["path"][0]]
        }
        return pred

    train_pred = process_predictions(pred_raw[0])
    val_pred = process_predictions(pred_raw[1])

    torch.save({"train": train_pred, "val": val_pred}, "ctp_feature.pt")


if __name__ == "__main__":
    main()
