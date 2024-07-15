import os
import logging
from os.path import join as opj
from functools import partial
from PIL import Image
import pandas as pd
import einops
import numpy as np
import torch

from tqdm import tqdm


def main():
    out_root = "/nfs/mm-isilon/brainscans/dropbox/exp/root_foundation_embs"
    slide_csv = "/nfs/turbo/umms-tocho/data/data_splits/he_neuro/all_he_slides.csv"

    slides = pd.read_csv(slide_csv)
    for m in ["uni", "gigapath", "conch", "plip", "virchow"]:
    #for m in ["virchow"]:
        ok = 0
        notok = []
        for i, ser in tqdm(slides.iterrows()):

            out_dir = opj(out_root, ser["patient"], ser['mosaic'])
            out_name = f"{ser['patient']}-{ser['mosaic']}-embs-{m}.dat"

            if os.path.exists(opj(out_dir, out_name)):
                ok += 1
            else:
                notok.append(i)
        print(f"{m} - {ok} / {len(slides)}")
        #for i in torch.split(torch.tensor(notok), 100):
        #    print(i.tolist())



if __name__ == "__main__":
    main()
