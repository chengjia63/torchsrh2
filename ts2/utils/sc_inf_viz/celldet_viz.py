from os.path import join as opj
import json
import itertools
import tqdm
import pandas as pd
from typing import List
import numpy as np
import os
import copy
import torch
import einops
import yaml
from collections import namedtuple
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter
from torchvision.transforms.functional import adjust_contrast, adjust_brightness
from ts2.data.transforms import HistologyTransform
from sklearn.manifold import TSNE

import altair as alt
from PIL import Image
import io
import base64

import torch
from transformers import AutoImageProcessor, AutoModel


def normalize_im(x):
    return (adjust_brightness(adjust_contrast(x, 2), 2) * 255).to(torch.uint8)


def rearrange_viz(x):
    return einops.rearrange(x, "c h w -> h w c")


def get_augs():
    hist_xform_params = yaml.safe_load("""
        which_set: srh
        base_aug_params:
            laser_noise_config: null
            get_third_channel_params:
                mode: three_channels
                subtracted_base: 0.07629394531
        strong_aug_params:
            aug_list:
            - which: center_crop_always_apply
              params: 
                size: 24
            aug_prob: 1
        """)
    return HistologyTransform(**hist_xform_params)


def get_dinov2_model():
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    model = AutoModel.from_pretrained('facebook/dinov2-base')
    model = model.cuda()
    return processor, model


def get_im(slide, scsrh_repl_root, k=32):
    slide_id = "_".join([slide["patient"], slide["mosaic"]])

    meta_fname = opj(scsrh_repl_root, f"{slide_id}_meta.json")

    with open(meta_fname) as fd:
        meta_s = json.load(fd)

    celltypes = [ms["celltype"] for ms in meta_s["cells"]]
    is_edge = torch.tensor([ms["is_edge"] for ms in meta_s["cells"]])
    is_nuclei = torch.tensor([c == "nuclei" for c in celltypes])
    is_mp = torch.tensor([c == "macrophage" for c in celltypes])

    nuclei_idx = torch.where(is_nuclei & ~is_edge)[0]
    nuclei_idx = nuclei_idx[torch.randperm(len(nuclei_idx))[:k]].tolist()
    mp_idx = torch.where(is_mp & ~is_edge)[0]
    mp_idx = mp_idx[torch.randperm(len(mp_idx))[:k]].tolist()

    mmap_fname = opj(scsrh_repl_root, f"{slide_id}_cells.dat")

    fd = np.memmap(mmap_fname,
                   dtype="uint16",
                   mode="r",
                   shape=tuple(meta_s["tensor_shape"]))

    nu_im = np.array(fd[nuclei_idx], dtype=float)
    mp_im = np.array(fd[mp_idx], dtype=float)

    fd._mmap.close()
    del fd

    nu_im = torch.from_numpy(nu_im).to(torch.float32).contiguous()
    mp_im = torch.from_numpy(mp_im).to(torch.float32).contiguous()

    nu_bg_mask = (1 - nu_im[:, -1, ...]).to(bool).unsqueeze(1).repeat(
        1, 2, 1, 1)
    nu_bg = all_class_mean["all"]["nu"][:2, ...].mean(
        dim=1, keepdim=True).mean(dim=2, keepdim=True).unsqueeze(0).repeat(
            nu_im.shape[0], 1, nu_im.shape[2], nu_im.shape[3])
    nu_im[:, :2, ...][nu_bg_mask] = nu_bg[nu_bg_mask]

    mp_bg_mask = (1 - mp_im[:, -1, ...]).to(bool).unsqueeze(1).repeat(
        1, 2, 1, 1)
    mp_bg = all_class_mean["all"]["mp"][:2, ...].mean(
        dim=1, keepdim=True).mean(dim=2, keepdim=True).unsqueeze(0).repeat(
            mp_im.shape[0], 1, mp_im.shape[2], mp_im.shape[3])
    mp_im[:, :2, ...][mp_bg_mask] = mp_bg[mp_bg_mask]
    import pdb; pdb.set_trace()
    return {
        "nu_im": nu_im,
        "mp_im": mp_im,
        "slide": f"{slide['patient']}.{slide['mosaic']}",
        "label": slide["label"]
    }


def inference_cells(data_s, aug, processor, model):

    def inference_one_set(x):
        if len(x) == 0:
            return torch.empty(0, 768)
        else:
            im = torch.stack([aug(i[:2, ...]) for i in x])
            inputs = processor(images=im,
                               return_tensors="pt",
                               do_rescale=False)["pixel_values"]
            emb = model(pixel_values=inputs.cuda())[1].detach().cpu()
            return emb

    data_s.update({
        "nu_emb": inference_one_set(data_s["nu_im"]),
        "mp_emb": inference_one_set(data_s["mp_im"]),
    })
    return data_s


def im_to_bytestr(image: Image) -> str:
    output = io.BytesIO()
    image.save(output, format='JPEG')
    return "data:image/jpeg;base64," + base64.b64encode(
        output.getvalue()).decode()


def make_save_chart(nu_df, color_map, outname: str):

    class_selection = alt.selection_point(fields=["label"], bind="legend")

    nu_chart = alt.Chart(nu_df
    ).encode(
        x=alt.X(
            "x",
            scale=alt.Scale(domain=[-1,1]),
            axis=alt.Axis(
                tickSize=0,
                values=[-1, -0.6, -0.2, 0.2, 0.6, 1],
                domain=False,
                labels=False,
                title=""
            )
        ),
        y=alt.Y(
            "y",
            scale=alt.Scale(domain=[-1,1]),
            axis=alt.Axis(
                tickSize=0,
                values=[-1, -0.6, -0.2, 0.2, 0.6, 1],
                domain=False,
                labels=False,
                title=""
            )
        ),
        size=alt.value(64),
        opacity=alt.condition(
            class_selection,
            alt.value(0.8),
            alt.value(0.1)
        ),
        color= alt.condition(
            class_selection,
            alt.Color(
                "label:N",
                scale=alt.Scale(
                    domain=list(color_map.keys()),
                    range=list(color_map.values())
                ),
            ),
            alt.value('lightgray')
        ),
        tooltip=["image", "slide"]
    ).mark_point(filled=True,
    ).add_params(class_selection)  # yapf: disable

    nu_chart_interactive = nu_chart.properties(
        width=1024,
        height=1024
    ).interactive()
    nu_chart_interactive.save(f"{outname}.json")
    nu_chart_interactive.save(f"{outname}.html") # yapf: disable


    nu_chart_small = nu_chart.encode(
        size=alt.value(16),
        color=alt.Color(
            "label:N",
            scale=alt.Scale(
                domain=list(color_map.keys()),
                range=list(color_map.values())
            ),
            legend=None
        )
    ).properties(width=512, height=512) # yapf: disable
    nu_chart_small.save(f"{outname}.pdf")
    nu_chart_small.save(f"{outname}.png")


all_class_mean = torch.load("all_tt_means.pt")


def main():
    scsrh_repl_root = "/nfs/turbo/umms-tocho-snr/exp/chengjia/scsrh_repl_root2/"

    train_slides = pd.read_csv(
        "/nfs/turbo/umms-tocho/data/data_splits/srh7v1/srh7v1_train.csv",
        dtype=str)

    test_slides = pd.read_csv(
        "/nfs/turbo/umms-tocho/data/data_splits/srh7v1/srh7v1_test.csv",
        dtype=str)

    color_map = {
        "hgg": "#fa5252",
        "lgg": "#fd7e14",
        "mening": "#fab005",
        "metast": "#82c91e",
        "pituita": "#15aabf",
        "schwan": "#7950f2",
        "normal": "#be4bdb"
    }

    slide_sample_per_class = 32
    cell_sample_per_slide = 32
    aug = get_augs()
    processor, model = get_dinov2_model()

    sampled_train_slides = train_slides.groupby("label").sample(
        slide_sample_per_class).reset_index()

    slides = [
        inference_cells(
            get_im(r, scsrh_repl_root=scsrh_repl_root,
                   k=cell_sample_per_slide), aug, processor, model)
        for i, r in tqdm.tqdm(sampled_train_slides.iterrows())
    ]

    def normalize_tsne_out(xy):
        min_vals = xy.min(axis=0)
        max_vals = xy.max(axis=0)
        xy_ = (xy - min_vals) / (max_vals - min_vals)
        return xy_ * 1.8 - 0.9

    tsne = TSNE(n_components=2)
    nu_xy = normalize_tsne_out(
        tsne.fit_transform(torch.cat([s["nu_emb"] for s in slides])))
    mp_xy = normalize_tsne_out(
        tsne.fit_transform(torch.cat([s["mp_emb"] for s in slides])))

    def list_of_dict_get_key_with_repkey(x, key, rep_key):
        return list(itertools.chain(*[[s[key]] * len(s[rep_key]) for s in x]))

    def tsne_embed_im(x, aug):
        return im_to_bytestr(
            Image.fromarray(
                rearrange_viz(normalize_im(aug(x[:2, ...]))).numpy()))

    make_save_chart(pd.DataFrame({
        "x":
        nu_xy[:, 0],
        "y":
        nu_xy[:, 1],
        "image": [tsne_embed_im(i, aug) for s in slides for i in s["nu_im"]],
        "label":
        list_of_dict_get_key_with_repkey(slides, "label", "nu_im"),
        "slide":
        list_of_dict_get_key_with_repkey(slides, "slide", "nu_im"),
    }),
                    color_map=color_map,
                    outname="train_nu_24")

    make_save_chart(pd.DataFrame({
        "x":
        mp_xy[:, 0],
        "y":
        mp_xy[:, 1],
        "image": [tsne_embed_im(i, aug) for s in slides for i in s["mp_im"]],
        "label":
        list_of_dict_get_key_with_repkey(slides, "label", "mp_im"),
        "slide":
        list_of_dict_get_key_with_repkey(slides, "slide", "mp_im")
    }),
                    color_map=color_map,
                    outname="train_mp_24")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
