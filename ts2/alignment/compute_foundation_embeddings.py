from os.path import join as opj
from functools import partial
from PIL import Image

import torch
import torchvision
from torchvision import transforms
import pytorch_lightning as pl
import einops

from transformers import AutoImageProcessor, AutoModel
from transformers import CLIPProcessor, CLIPModel

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked

from conch.open_clip_custom import create_model_from_pretrained as create_conch_model

from ts2.models.ssl import instantiate_backbone
from ts2.models.hipt.hipt_model_utils import get_vit256 as get_hipt_bbone
from ts2.data.histology_data_module import PatchDataModule
from ts2.train.infra import (parse_args, read_process_cf, setup_infra_training,
                             setup_infra_testing, get_rank)


class PLIPEvalSystem(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.processor = CLIPProcessor.from_pretrained("vinid/plip")
        self.model = CLIPModel.from_pretrained("vinid/plip")

    @torch.inference_mode()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        raw_im = (batch["image"].squeeze() * 255).to(torch.uint8)
        proc_im = self.processor.image_processor(raw_im)
        proc_im = torch.stack(
            [torch.tensor(i) for i in proc_im["pixel_values"]])

        out = self.model.get_image_features(proc_im.to(
            self.model.device)).detach()

        return {
            "path": batch["path"],
            "label": batch["label"],
            "embeddings": out
        }


class UNIEvalSystem(pl.LightningModule):

    def __init__(self, ckpt_path):
        super().__init__()
        self.model = timm.create_model("vit_large_patch16_224",
                                       img_size=224,
                                       patch_size=16,
                                       init_values=1e-5,
                                       num_classes=0,
                                       dynamic_img_size=True)
        self.model.load_state_dict(torch.load(ckpt_path, map_location="cpu"),
                                   strict=True)

        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize(224),
            #transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])

    @torch.inference_mode()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        raw_im = (batch["image"].squeeze() * 255)
        image = self.transform(raw_im)
        out = self.model(image.to(batch["image"].device))
        return {
            "path": batch["path"],
            "label": batch["label"],
            "embeddings": out
        }


class ConchEvalSystem(pl.LightningModule):

    def __init__(self, ckpt_path):
        super().__init__()
        self.model, self.preprocess = create_conch_model(
            "conch_ViT-B-16", ckpt_path)
        self.model.eval()

    @torch.inference_mode()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        raw_im = (batch["image"].squeeze() * 255).to(torch.uint8)
        raw_im = [
            Image.fromarray(
                einops.rearrange(i.detach().cpu().numpy(), "c h w -> h w c"))
            for i in raw_im
        ]
        proc_im = torch.stack([self.preprocess(ri) for ri in raw_im])
        out = self.model.encode_image(proc_im.to(batch["image"].device),
                                      proj_contrast=False,
                                      normalize=False)

        return {
            "path": batch["path"],
            "label": batch["label"],
            "embeddings": out
        }


class VirchowEvalSystem(pl.LightningModule):

    def __init__(self, ckpt_path):
        super().__init__()
        self.model = timm.create_model("vit_huge_patch14_224",
                                       img_size=224,
                                       init_values=1e-5,
                                       num_classes=0,
                                       mlp_ratio=5.3375,
                                       global_pool="",
                                       dynamic_img_size=True,
                                       mlp_layer=SwiGLUPacked,
                                       act_layer=torch.nn.SiLU)
        self.model.load_state_dict(torch.load(ckpt_path))

        self.model = self.model.eval()

        self.model.pretrained_cfg = {
            "tag": "virchow_v1",
            "custom_load": False,
            "input_size": [3, 224, 224],
            "fixed_input_size": False,
            "interpolation": "bicubic",
            "crop_pct": 1.0,
            "crop_mode": "center",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "num_classes": 0,
            "pool_size": None,
            "first_conv": "patch_embed.proj",
            "classifier": "head",
            "license": "CC-BY-NC-ND-4.0"
        }

        self.transforms = create_transform(
            **resolve_data_config(self.model.pretrained_cfg, model=self.model))

    @torch.inference_mode()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        raw_im = (batch["image"].squeeze() * 255).to(torch.uint8)
        raw_im = [
            Image.fromarray(
                einops.rearrange(i.detach().cpu().numpy(), "c h w -> h w c"))
            for i in raw_im
        ]
        proc_im = torch.stack([self.transforms(ri) for ri in raw_im])
        output = self.model(proc_im.to(batch["image"].device))

        class_token = output[:, 0]
        #patch_tokens = output[:, 1:]

        #embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)

        return {
            "path": batch["path"],
            "label": batch["label"],
            "embeddings": class_token  # embedding
        }


class GigapathEvalSystem(pl.LightningModule):

    def __init__(self, ckpt_path):
        super().__init__()

        self.model = timm.create_model("vit_giant_patch14_dinov2",
                                       global_pool="token",
                                       img_size=224,
                                       in_chans=3,
                                       patch_size=16,
                                       embed_dim=1536,
                                       depth=40,
                                       num_heads=24,
                                       init_values=1e-05,
                                       mlp_ratio=5.33334,
                                       num_classes=0)
        self.model.load_state_dict(torch.load(ckpt_path, map_location="cpu"),
                                   strict=True)

        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(
                256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])

    @torch.inference_mode()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        raw_im = (batch["image"].squeeze() * 255)
        image = self.transform(raw_im)
        out = self.model(image.to(batch["image"].device))
        return {
            "path": batch["path"],
            "label": batch["label"],
            "embeddings": out
        }


class ImageNetResnetEvalSystem(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet34(weights="IMAGENET1K_V1")
        #self.model = torchvision.models.resnet50(weights="IMAGENET1K_V2")
        #self.model = torchvision.models.resnext50_32x4d(weights="IMAGENET1K_V2")
        #self.model = torchvision.models.resnext101_64x4d(weights="IMAGENET1K_V1")

        self.model.fc = torch.nn.Identity()
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])

    @torch.inference_mode()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        raw_im = (batch["image"].squeeze() * 255)
        image = self.transform(raw_im)
        out = self.model(image.to(batch["image"].device))
        return {
            "path": batch["path"],
            "label": batch["label"],
            "embeddings": out
        }


class DINOv2EvalSystem(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(
            'facebook/dinov2-base')
        self.model = AutoModel.from_pretrained('facebook/dinov2-base')

    @torch.inference_mode()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        raw_im = (batch["image"].squeeze() * 255).to(torch.uint8)
        raw_im = [
            Image.fromarray(
                einops.rearrange(i.detach().cpu().numpy(), "c h w -> h w c"))
            for i in raw_im
        ]

        proc_im = torch.stack([
            self.processor(ri, return_tensors="pt")["pixel_values"][0]
            for ri in raw_im
        ])
        output = self.model(pixel_values=proc_im.to(batch["image"].device))

        return {
            "path": batch["path"],
            "label": batch["label"],
            "embeddings": output.last_hidden_state[:, 0, :]
        }


class CLIPEvalSystem(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-large-patch14")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")

    @torch.inference_mode()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        raw_im = (batch["image"].squeeze() * 255).to(torch.uint8)
        proc_im = self.processor.image_processor(raw_im)
        proc_im = torch.stack(
            [torch.tensor(i) for i in proc_im["pixel_values"]])

        out = self.model.get_image_features(proc_im.to(
            self.model.device)).detach()

        return {
            "path": batch["path"],
            "label": batch["label"],
            "embeddings": out
        }


class HIPTEvalSystem(pl.LightningModule):

    def __init__(self, ckpt_path):
        super().__init__()
        self.model = get_hipt_bbone(pretrained_weights=ckpt_path)
        self.model.eval()

        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        self.transform = transforms.Normalize(mean=mean, std=std)

    @torch.inference_mode()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        out = self.model.forward(self.transform(batch["image"].squeeze()))
        return {
            "path": batch["path"],
            "label": batch["label"],
            "embeddings": out
        }


class InDomainEvalSystem(pl.LightningModule):

    def __init__(self, ckpt_path):
        super().__init__()
        self.model = instantiate_backbone(which="resnet50", params={})

        state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        state_dict = {
            k.removeprefix("model.bb."): state_dict[k]
            for k in state_dict if k.startswith("model.bb")
        }

        self.model.load_state_dict(state_dict, strict=True)

        self.model.eval()

    @torch.inference_mode()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        out = self.model(batch["image"].squeeze()).detach().cpu()
        return {
            "path": batch["path"],
            "label": batch["label"],
            "embeddings": out
        }


#class TODOEvalSystem(pl.LightningModule):
#    def __init__(self):
#        super().__init__()
#
#    @torch.inference_mode()
#    def predict_step(self, batch, batch_idx, dataloader_idx=0):
#        pass


def process_predictions(predictions):
    pred = {
        "embeddings": torch.cat([p["embeddings"] for p in predictions]),
        "label": torch.cat([p["label"] for p in predictions]),
        "path": [pk for p in predictions for pk in p["path"][0]]
    }
    return pred


@torch.inference_mode()
def main():

    cf = read_process_cf(parse_args())
    dm = PatchDataModule(config=cf)

    make_model = lambda x: opj(cf["testing"]["model_root"], x)

    models = {
        #"uni":
        #partial(
        #    UNIEvalSystem,
        #    ckpt_path=make_model(
        #        "uni/vit_large_patch16_224.dinov2.uni_mass100k/pytorch_model.bin"
        #    )),
        #"plip":
        #PLIPEvalSystem,
        #"conch":
        #partial(ConchEvalSystem,
        #        ckpt_path=make_model("conch/pytorch_model.bin")),
        #"virchow":
        #partial(VirchowEvalSystem,
        #        ckpt_path=make_model("virchow/pytorch_model.bin")),
        #"gigapath":
        #partial(GigapathEvalSystem,
        #        ckpt_path=make_model("gigapath/pytorch_model.bin")),
        #"imresnet":
        #ImageNetResnetEvalSystem,
        #"dinov2":
        #DINOv2EvalSystem,
        #"clip":
        #CLIPEvalSystem,
        #"hipt":
        #partial(HIPTEvalSystem, make_model("hipt/vit256_small_dino.pth")),
        "id_simclr":
        partial(
            InDomainEvalSystem,
            ckpt_path=
            "/nfs/turbo/umms-tocho/models/hidisc_tcga/435f56fa_simclr_1000.ckpt"
        ),
        "id_hidisc":
        partial(
            InDomainEvalSystem,
            ckpt_path=
            "/nfs/turbo/umms-tocho/models/hidisc_tcga/e591e086_hidisc.slide_1000.ckpt"
        )
    }

    for which_model in models.keys():
        model = models[which_model]()
        pred_trainer = pl.Trainer(accelerator="gpu",
                                  devices=1,
                                  default_root_dir=".",
                                  inference_mode=True)  # deterministic=True)

        pred_raw = pred_trainer.predict(model, datamodule=dm)

        train_pred = process_predictions(pred_raw[0])
        val_pred = process_predictions(pred_raw[1])

        torch.save({
            "train": train_pred,
            "val": val_pred
        }, f"{which_model}_feature.pt")


if __name__ == "__main__":
    main()
