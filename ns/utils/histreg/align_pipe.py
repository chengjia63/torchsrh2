import os
from os.path import join as opj
import argparse
import logging

import openslide
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
from tqdm import tqdm
from typing import List
import yaml
import cv2

from PIL import Image
import einops
import re

import tifffile
import torch
import torch.nn.functional as F
from itertools import chain, accumulate
import pickle

from histreg.superglue import match_superglue, basic_preprocessing, default_match_cf, np_to_tensor, inverse_affine_transform, make3x3
from histreg.bf import match_bf


def sanitize_string(string):
    return re.sub(r'[^a-zA-Z0-9]', '', string)


def get_thumbnail(slide, target_downsample=64):
    tn_level = 2
    tn_downsample = slide.level_downsamples[tn_level]
    thumb = slide.read_region(location=(0, 0),
                              level=tn_level,
                              size=slide.level_dimensions[tn_level])
    if not (int(tn_downsample) == target_downsample):
        scale_factor = tn_downsample / target_downsample
        original_width, original_height = thumb.size
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        thumb = thumb.resize((new_width, new_height))
        tn_downsample = target_downsample

    return np.array(thumb.convert('RGB')), tn_downsample


def get_sections_annot(he_annot):
    annot_mask = np.all(he_annot == np.array([[[0, 255, 0]]]), axis=2)

    contours, _ = cv2.findContours(annot_mask.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return [None], [None]

    # Create a filled mask for each contour
    masked_ims = []
    for i, contour in enumerate(contours):
        # Create an empty mask
        mask = np.zeros_like(annot_mask, dtype=float)
        # Fill the contour
        cv2.drawContours(mask, [contour], -1, 1, thickness=cv2.FILLED)
        masked_ims.append(torch.tensor(mask))

    if len(masked_ims) == 1:
        neg_mask = ~masked_ims[0].to(bool)
    else:
        neg_mask = ~torch.stack(masked_ims).any(dim=0)

    return masked_ims, [neg_mask] * len(masked_ims)


#def circular_kernel(radius):
#    size = 2 * radius + 1
#    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
#    center = radius
#    dist = (x - center) ** 2 + (y - center) ** 2
#    kernel = (dist <= radius ** 2).float()
#    return kernel
#
## Function to compute ring coordinates
#def ring_coordinates(binary_mask, dilation_radius=5):
#    # Ensure binary mask is float for calculations
#    binary_mask = binary_mask.float()
#
#    # Generate circular kernel
#    kernel = circular_kernel(dilation_radius).unsqueeze(0).unsqueeze(0)
#
#    # Dilate the mask
#    dilated_mask = F.conv2d(binary_mask.unsqueeze(0).unsqueeze(0), kernel, padding=dilation_radius).squeeze()
#    dilated_mask = (dilated_mask > 0).float()
#
#    # Create the ring mask
#    ring_mask = dilated_mask - binary_mask
#    ring_mask = (ring_mask > 0).float()
#
#    # Get coordinates where the ring mask is 1
#    ring_coords = torch.nonzero(ring_mask, as_tuple=False)
#    return ring_coords


def interpolate_crop_mask(he_proc_shape, he_mask):
    desired_mask_size = max(he_proc_shape[0], he_proc_shape[1])
    ann_size = max(he_mask.shape[0], he_mask.shape[1])

    if desired_mask_size < ann_size:
        image = he_mask[:he_proc_shape[0], :he_proc_shape[1]].to(torch.uint8)
        #canvas = torch.zeros_like(he_mask).to(torch.uint8)
        return image
    else:
        # using nearest because this is a binary mask
        new_size = (desired_mask_size, desired_mask_size)
        mask_resized = F.interpolate(he_mask.to(
            torch.uint8).unsqueeze(0).unsqueeze(0),
                                     size=new_size,
                                     mode='nearest').squeeze(0).squeeze(0)
        if he_proc_shape[0] < he_proc_shape[1]:
            mask_resized = mask_resized[:he_proc_shape[0], :]
        else:
            mask_resized = mask_resized[:, :he_proc_shape[1]]
        return mask_resized


def rescale_and_mask(he_proc, he_mask, he_mask_neg):
    if he_mask is None:
        if len(he_proc.shape) == 3:
            means = None
        else:
            means = he_proc.mean()
        return he_proc, means, np.eye(3)

    mask_resized = interpolate_crop_mask(he_proc.shape, he_mask)
    fg_coords2 = torch.argwhere(mask_resized.to(bool))
    min_fg = fg_coords2.min(dim=0).values  # r,c
    max_fg = fg_coords2.max(dim=0).values  # r,c
    mask_resized = mask_resized[min_fg[0]:max_fg[0], min_fg[1]:max_fg[1]]
    xform = np.eye(3)
    xform[0:2, 2] = [-min_fg[1], -min_fg[0]]

    neg_mask_resized = interpolate_crop_mask(he_proc.shape, he_mask_neg)
    bg_coords2 = torch.argwhere(neg_mask_resized.to(bool))

    if len(he_proc.shape) == 3:
        if he_proc.shape[-1] == 3:  # HWC
            fg = he_proc[min_fg[0]:max_fg[0], min_fg[1]:max_fg[1], :]
            mask3 = einops.repeat(mask_resized, "H W -> H W C", C=3)
            bg_avg = he_proc[bg_coords2[:, 0], bg_coords2[:,
                                                          1], :].mean(axis=0)

        elif he_proc.shape[0] == 3:  # CHW
            fg = he_proc[:, min_fg[0]:max_fg[0], min_fg[1]:max_fg[1]]
            mask3 = einops.repeat(mask_resized, "H W -> C H W", C=3)
            bg_avg = he_proc[:, bg_coords2[:, 0], bg_coords2[:,
                                                             1]].mean(axis=0)

        else:
            raise ValueError("3 channel he_proc must be CHW or HWC, C=3")

        masked_im = mask3 * fg + (1 - mask3) * einops.repeat(
            bg_avg, "C -> H W C", H=fg.shape[0], W=fg.shape[1])
        bg_avg = None

    else:
        fg = he_proc[min_fg[0]:max_fg[0], min_fg[1]:max_fg[1]]
        bg_avg = he_proc[bg_coords2[:, 0], bg_coords2[:, 1]].mean()
        masked_im = mask_resized * fg + (1 - mask_resized) * einops.repeat(
            [bg_avg], "1 -> H W", H=fg.shape[0], W=fg.shape[1])

    return masked_im, bg_avg, xform


def inverse_affine_transform_3x3(T):
    """
    Inverse of a 3x3 affine transformation matrix.
    
    Parameters:
    T (np.ndarray): 3x3 affine transformation matrix.
    
    Returns:
    T_inv (np.ndarray): 3x3 inverse affine transformation matrix.
    """
    if T.shape != (3, 3):
        raise ValueError("Input matrix T must be a 3x3 matrix.")

    # Extract the top-left 2x2 matrix and the translation vector
    A = T[:2, :2]
    b = T[:2, 2]

    # Check if A is invertible
    if np.linalg.det(A) == 0:
        raise ValueError("The linear transformation matrix A is singular")

    # Invert A
    A_inv = np.linalg.inv(A)

    # Compute the new translation vector
    b_inv = -np.dot(A_inv, b)

    # Construct the inverse transformation matrix
    T_inv = np.eye(3)
    T_inv[:2, :2] = A_inv
    T_inv[:2, 2] = b_inv

    return T_inv


def align_block(mask_annot, mask_annot_meta, match_cf, svs_root, out_dir):
    #mask_annot_meta = mask_annot_meta.rename(
    #    columns={"Unnamed: 0": "main_idx"})

    if len(mask_annot.shape) == 3: # 1 physical slide
        mask_annot = np.expand_dims(mask_annot, 0)

    section_mask_both = [get_sections_annot(m) for m in mask_annot]
    section_mask = [smb[0] for smb in section_mask_both]
    section_mask_neg = [smb[1] for smb in section_mask_both]

    flat_mask = list(chain.from_iterable(section_mask))
    flat_mask_neg = list(chain.from_iterable(section_mask_neg))
    lengths = list(map(len, section_mask))  # Lengths of each sublist
    starts = [0] + list(accumulate(lengths[:-1]))  # Start idx of each sublist
    list_indices = [
        list(range(start, start + length))
        for start, length in zip(starts, lengths)
    ]

    mask_annot_meta["sections_mask"] = list_indices
    curr_block = mask_annot_meta.explode("sections_mask").reset_index(
        drop=True)
    curr_block = curr_block[~(curr_block["comment"] == "RM")].reset_index(
        drop=True)
    #curr_block = curr_block.iloc[[0,3,5,6,7,-1]].reset_index(drop=True) # for debugging
    #curr_block = curr_block.iloc[[0,3]].reset_index(drop=True) # for debugging

    assert curr_block['UM Accession'].nunique() == 1
    assert curr_block['Block'].nunique() == 1
    curr_block_str = ".".join(
        (curr_block["UM Accession"][0], curr_block["Block"][0]))

    curr_he = curr_block[curr_block["Stain"] == "H&E"]
    curr_ihc = curr_block[~(curr_block["Stain"] == "H&E")]

    if len(curr_he) == 0:
        logging.info(f"@@@ {curr_block_str},no_he,abort")
        return

    if len(curr_block) == 1:
        logging.info(f"@@@ {curr_block_str},one_section_only,ok")
        return

    logging.info("Num slide: HE / IHC / Total: " +
                 f"{len(curr_he)} / {len(curr_ihc)} / {len(curr_block)}")

    matrices = np.zeros((len(curr_he), len(curr_block), 3, 3))
    num_matches = np.zeros((len(curr_he), len(curr_block)))
    align_viz_binary = [[None for _ in range(len(curr_block))]
                        for _ in range(len(curr_he))]
    align_viz_im = [[None for _ in range(len(curr_block))]
                    for _ in range(len(curr_he))]

    he_binary = []
    he_im = []
    he_scale = []

    all_binary = []
    all_im = []
    all_scale = []

    for he_i, he_s in curr_he.iterrows():
        he_slide = openslide.OpenSlide(opj(svs_root, he_s["path"]))
        he_thumbnail_orig, he_scale_i = get_thumbnail(he_slide)
        src_is_he = (he_s["Stain"] == "H&E")
        he_proc_orig = basic_preprocessing(he_thumbnail_orig,
                                           red_only=src_is_he)

        he_thumbnail, _, _ = rescale_and_mask(
            he_thumbnail_orig, flat_mask[he_s["sections_mask"]],
            flat_mask_neg[he_s["sections_mask"]])
        he_proc, he_proc_bg, he_xform = rescale_and_mask(
            he_proc_orig, flat_mask[he_s["sections_mask"]],
            flat_mask_neg[he_s["sections_mask"]])

        he_binary.append(np.array(he_proc).squeeze().astype(np.float64))
        he_im.append(np.array(he_thumbnail).squeeze().astype(np.float64))
        he_scale.append(he_scale_i)

        for ihc_i, ihc_s in curr_block.iterrows():
            ihc_slide = openslide.OpenSlide(opj(svs_root, ihc_s["path"]))
            ihc_thumbnail_orig, ihc_scale_j = get_thumbnail(ihc_slide)
            tgt_is_he = (ihc_s["Stain"] == "H&E")
            ihc_proc_orig = basic_preprocessing(ihc_thumbnail_orig,
                                                red_only=tgt_is_he)
            logging.info(f"{he_s['Stain']} {he_i} -> {ihc_s['Stain']} {ihc_i}")

            ihc_thumbnail, _, _ = rescale_and_mask(
                ihc_thumbnail_orig, flat_mask[ihc_s["sections_mask"]],
                flat_mask_neg[ihc_s["sections_mask"]])
            ihc_proc, _, ihc_xform = rescale_and_mask(
                ihc_proc_orig, flat_mask[ihc_s["sections_mask"]],
                flat_mask_neg[ihc_s["sections_mask"]])

            if he_i == 0:
                all_binary.append(
                    np.array(ihc_proc).squeeze().astype(np.float64))
                all_im.append(
                    np.array(ihc_thumbnail).squeeze().astype(np.float64))
                all_scale.append(ihc_scale_j)

            if match_cf.get("bf_only", False):
                M, nm = match_bf(he_proc.to(torch.float), ihc_proc.to(torch.float))
                logging.info(f"{he_s['Stain']} {he_i} -> {ihc_s['Stain']} {ihc_i}: {nm} BF dice")
            else:
                M, nm = match_superglue(he_proc.to(torch.float),
                                    ihc_proc.to(torch.float),
                                        match_cf,
                                        he_proc_bg,
                                        angle_step=20)

                #if nm < 50:  # do BF matching - it is too slow, will do it later
                #    M, nm = match_bf(he_proc.to(torch.float), ihc_proc.to(torch.float))
                #    logging.info(f"{he_s['Stain']} {he_i} -> {ihc_s['Stain']} {ihc_i}: {nm} BF dice")
                #else:

                logging.info(f"{he_s['Stain']} {he_i} -> {ihc_s['Stain']} {ihc_i}: {nm} matches")

            M = make3x3(inverse_affine_transform(
                ihc_xform[:2, ...])) @ M @ he_xform

            matrices[he_i, ihc_i, ...] = M
            num_matches[he_i, ihc_i] = nm
            transformed1 = cv2.warpAffine(
                np.array(he_proc_orig).squeeze().astype(np.float64),
                M[:2, ...].squeeze(),
                np.array(ihc_proc_orig).squeeze().shape[::-1])
            b = np.zeros(transformed1.shape)
            stacked_im = np.dstack(
                (transformed1,
                 np.array(ihc_proc_orig).squeeze().astype(np.float64), b))

            transformed2 = cv2.warpAffine(
                np.array(he_thumbnail_orig).squeeze().astype(np.float64),
                M[:2, ...].squeeze(),
                np.array(ihc_proc_orig).squeeze().shape[::-1])
            overlay = cv2.addWeighted(
                transformed2, .2,
                np.array(ihc_thumbnail_orig).squeeze().astype(np.float64), .8,
                0).astype(np.uint8)

            align_viz_binary[he_i][ihc_i] = stacked_im
            align_viz_im[he_i][ihc_i] = overlay

    with open(opj(out_dir, f"{curr_block_str}_align.pkl"), "wb") as fd:
        pickle.dump(
            {
                "matrices": matrices,
                "num_matches": num_matches,
                "all_scale": all_scale
            }, fd)

    with open(opj(out_dir, f"{curr_block_str}_align_viz.pkl"), "wb") as fd:
        pickle.dump(
            {
                "align_viz_binary": align_viz_binary,
                "align_viz_im": align_viz_im,
                "he_binary": he_binary,
                "he_im": he_im,
                "he_scale": he_scale,
                "all_binary": all_binary,
                "all_im": all_im
            }, fd)

    plt.rcParams['pdf.fonttype'] = 42

    fig, axs = plt.subplots(len(align_viz_binary) + 1,
                            len(align_viz_binary[0]) + 1,
                            figsize=(20, 5),
                            dpi=300)
    for i, ims_i in enumerate(align_viz_im):
        for j, im_ij in enumerate(ims_i):
            axs[i + 1][j + 1].imshow(im_ij)
            axs[i + 1][j + 1].axis("off")

    for im_i, (i, r) in zip(he_im, curr_he.iterrows()):
        axs[i + 1][0].imshow(im_i.astype(np.uint8))
        axs[i + 1][0].axis("off")
        axs[i + 1][0].text(x=-0.1,
                           y=0.5,
                           s=r["Stain"],
                           va='center',
                           ha='center',
                           rotation=90,
                           fontsize=plt.rcParams['axes.titlesize'],
                           transform=axs[i + 1][0].transAxes)
    for im_i, (i, r) in zip(all_im[:len(curr_block)], curr_block.iterrows()):
        axs[0][i + 1].imshow(im_i.astype(np.uint8))
        axs[0][i + 1].axis("off")
        axs[0][i + 1].set_title(r["Stain"])

    axs[0][0].axis("off")

    plt.tight_layout()
    plt.savefig(f"{out_dir}/{curr_block_str}_im_align.pdf")

    fig, axs = plt.subplots(len(align_viz_binary) + 1,
                            len(align_viz_binary[0]) + 1,
                            figsize=(20, 5),
                            dpi=300)
    for i, ims_i in enumerate(align_viz_binary):
        for j, im_ij in enumerate(ims_i):
            axs[i + 1][j + 1].imshow(im_ij)
            axs[i + 1][j + 1].axis("off")

    for im_i, (i, r) in zip(he_binary, curr_he.iterrows()):
        im = (einops.repeat(im_i, "h w -> h w c", c=3) * 255).astype(np.uint8)
        im[:, :, [1, 2]] = 0
        axs[i + 1][0].imshow(im)
        axs[i + 1][0].axis("off")
        axs[i + 1][0].text(x=-0.1,
                           y=0.5,
                           s=r["Stain"],
                           va='center',
                           ha='center',
                           rotation=90,
                           fontsize=plt.rcParams['axes.titlesize'],
                           transform=axs[i + 1][0].transAxes)

    for im_i, (i, r) in zip(all_binary[:len(curr_block)],
                            curr_block.iterrows()):
        im = (einops.repeat(im_i, "h w -> h w c", c=3) * 255).astype(np.uint8)
        im[:, :, [0, 2]] = 0
        axs[0][i + 1].imshow(im)
        axs[0][i + 1].axis("off")
        axs[0][i + 1].set_title(r["Stain"])

    axs[0][0].axis("off")

    plt.tight_layout()
    plt.savefig(opj(out_dir, f"{curr_block_str}_mask_align.pdf"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--config',
                        type=argparse.FileType('r'),
                        required=True,
                        help='config file for training')
    args = parser.parse_args()
    cf = yaml.safe_load(args.config)

    logging_format_str = "[%(levelname)-s|%(asctime)s|%(name)s|" + \
        "%(filename)s:%(lineno)d|%(funcName)s] %(message)s"
    logging.basicConfig(level=logging.INFO,
                        format=logging_format_str,
                        datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler()],
                        force=True)
    logging.info("Exp root {}".format(cf["infra"]["out_dir"]))
    logging.getLogger("fontTools.subset").setLevel(logging.WARNING)

    os.makedirs(cf["infra"]["out_dir"], exist_ok=True)

    env_var = dict(os.environ)

    if "SLURM_ARRAY_TASK_ID" in env_var:
        taskid = int(env_var["SLURM_ARRAY_TASK_ID"])
        logging.info("get taskid {}".format(taskid))
    else:
        taskid = 0
        logging.warning("default taskid {}".format(taskid))

    blocks_df = pd.read_csv(cf["infra"]["blocks_csv"])
    start_id = max(taskid * cf["infra"]["num_block_per_job"], 0)
    end_id = min((taskid + 1) * cf["infra"]["num_block_per_job"],
                 len(blocks_df))

    blocks = blocks_df.iloc[start_id:end_id]["block"].tolist()

    for b in tqdm(blocks):
        logging.info(f"START - {b}")
        try:
            mask_annot = tifffile.imread(
                opj(cf["infra"]["annot_root"], "block_annot", f"{b}.tiff"))
            mask_annot_meta = pd.read_csv(
                opj(cf["infra"]["annot_root"], "meta",
                    f"{b}_sections_annot_meta.csv"))
            align_block(mask_annot,
                        mask_annot_meta,
                        cf["alignment"],
                        svs_root=cf["infra"]["svs_root"],
                        out_dir=cf["infra"]["out_dir"])
            logging.info(f"OK - {b}")

        except Exception as e:
            logging.error(f"@@@ FAIL - {b}")
            logging.error(e.__class__.__name__)
            logging.error(e)

        #torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
