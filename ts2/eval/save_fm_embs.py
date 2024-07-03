import os
import logging
from os.path import join as opj
from functools import partial
from PIL import Image
import pandas as pd
import torch
import torchvision
from torchvision import transforms
import pytorch_lightning as pl
import einops
import numpy as np

from omegaconf import OmegaConf
from transformers import AutoImageProcessor, AutoModel
from transformers import CLIPProcessor, CLIPModel
import json

from ts2.train.infra import parse_args, read_process_cf
from ts2.train.main import instantiate_lightning_module
from ts2.data.db_improc import instantiate_process_read
from ts2.data.transforms import HistologyTransform

from tqdm import tqdm


class FoundationEmbeddingSaver():

    def __init__(self):

        self.cf = read_process_cf(parse_args())
        logging.info(self.cf)

        self.model = instantiate_lightning_module(
            which=self.cf.lightning_module.which,
            params=self.cf.lightning_module.params,
            training_params=None)
        self.model.cuda()

        self.prf = instantiate_process_read(**self.cf.data.process_read)
        self.slides = pd.read_csv(self.cf.data.infra.slide_csv)
        self.xform = HistologyTransform(**self.cf.data.transform_params)

    def __call__(self):
        for i in tqdm(range(self.cf.infra.inf_idx[0], self.cf.infra.inf_idx[1])):
            logging.info(
                f"{i} in {{{self.cf.infra.inf_idx[0]}..{self.cf.infra.inf_idx[1]}}}"
            )
            try:
                self.save_one_slide(ser=self.slides.iloc[i])
            except:
                logging.error(
                    f"@@@ FAILED {self.cf.lightning_module.tag} - {i}"
                )
                

    @torch.inference_mode()
    def save_one_slide(self, ser: pd.Series):
        logging.info(f"{ser['patient']}/{ser['mosaic']}")
        mmap_path = opj(self.cf.data.infra.data_root, ser["institution"],
                        ser["patient"], ser["mosaic"], "patches",
                        f"{ser['patient']}-{ser['mosaic']}-patches.dat")
        pt_meta_path = opj(self.cf.data.infra.data_root, ser["institution"],
                           ser["patient"], f"{ser['patient']}_meta.json")

        with open(pt_meta_path) as fd:
            pt_meta = json.load(fd)["slides"][ser["mosaic"]]

        tensor_shape = tuple(
            pt_meta["predictions"]["226232a4"]["tensor_shape"])
        slide_tensor = self.xform(
            self.prf(mmap_path, tensor_shape, np.arange(tensor_shape[0])))

        slide_emb = torch.cat([
            self.model.forward(i.cuda()) for i in 
                slide_tensor.split(self.cf.data.inference_batch_size)
        ]).detach().cpu().numpy()

        out_dir = opj(self.cf.infra.out_root, ser["patient"], ser['mosaic'])
        out_name = f"{ser['patient']}-{ser['mosaic']}-embs-{self.cf.lightning_module.tag}.dat"

        if not os.path.exists(out_dir): os.makedirs(out_dir)

        fd = np.memmap(opj(out_dir, out_name),
                       dtype="float32",
                       mode="w+",
                       shape=slide_emb.shape)
        fd[:] = slide_emb
        fd.flush()


def main():

    logging_format_str = "[%(levelname)-s|%(asctime)s|%(name)s|" + \
        "%(filename)s:%(lineno)d|%(funcName)s] %(message)s"
    logging.basicConfig(level=logging.INFO,
                        format=logging_format_str,
                        datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler()],
                        force=True)

    FoundationEmbeddingSaver()()

    #de/chengjia/torchsrh2/ts2/eval $ ls /nfs/mm-isilon/brainscans/dropbox/data/root_histology_db/he.plip/cptac/cptac_c3l_00016/21/patches/
    #cptac_c3l_00016-21-patches.dat

    #pred_trainer = pl.Trainer(accelerator="gpu",
    #                          devices=1,
    #                          default_root_dir=".",
    #                          inference_mode=True)  # deterministic=True)
    #
    #pred_raw = pred_trainer.predict(model, datamodule=dm)
    #
    #train_pred = process_predictions(pred_raw[0])
    #val_pred = process_predictions(pred_raw[1])
    #
    #torch.save({
    #    "train": train_pred,
    #    "val": val_pred
    #}, f"{which_model}_feature.pt")


if __name__ == "__main__":
    main()
