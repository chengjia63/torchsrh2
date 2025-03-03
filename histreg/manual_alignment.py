import pickle
from os.path import join as opj
import matplotlib.pyplot as plt
import numpy as np

from typing import Dict
from histreg.superglue import basic_preprocessing, np_to_tensor, inverse_affine_transform, make3x3
from align_pipe import sanitize_string, get_thumbnail, get_sections_annot, interpolate_crop_mask, rescale_and_mask, inverse_affine_transform_3x3
import tifffile
import pandas as pd
from itertools import chain, accumulate
import openslide
import logging
import cv2
import torch.nn.functional as F
import einops
import torch
import os

import numpy as np

def decompose_affine_matrix(matrix):
    """
    Decomposes a 2x3 or 3x3 affine transformation matrix into:
    rotation (theta in degrees), scale (sx, sy), shear (shear in degrees), translation (tx, ty).

    Args:
        matrix (numpy.ndarray): A 2x3 or 3x3 affine transformation matrix.

    Returns:
        dict: A dictionary containing 'rotation', 'scale_x', 'scale_y', 'shear', 'tx', 'ty'.
    """
    # Ensure matrix is 3x3
    if matrix.shape == (2, 3):
        matrix = np.vstack([matrix, [0, 0, 1]])
    elif matrix.shape != (3, 3):
        raise ValueError("Input matrix must be 2x3 or 3x3.")

    A = matrix[:2, :2]  # Extract the 2x2 transformation submatrix
    tx, ty = matrix[:2, 2]  # Extract translation components

    # Compute scale factors (remove shear effects)
    sx = np.linalg.norm(A[:, 0])  # Length of first column

    theta = np.arctan2(A[1, 0], A[0, 0])
    msy = A[0,1] * np.cos(theta) + A[1,1] * np.sin(theta)


    if np.sin(theta) == 0:
        sy = (A[1,1] - msy * np.sin(theta)) / np.cos(theta)
    else:
        sy = (msy * np.cos(theta) - A[0,1]) / np.sin(theta)

    shear = msy / sy

    return {
        "rotation": np.degrees(theta),
        "scale_x": sx,
        "scale_y": sy,
        "shear": np.degrees(shear),
        "tx": tx,
        "ty": ty
    }

def reconstruct_affine_matrix(params):
    """
    Reconstructs a 3x3 affine transformation matrix from:
    rotation (degrees), scale, shear (degrees), and translation.

    Args:
        params (dict): Dictionary containing 'rotation', 'scale_x', 'scale_y', 'shear', 'tx', 'ty'.

    Returns:
        numpy.ndarray: The reconstructed 3x3 affine transformation matrix.
    """
    theta = np.radians(params["rotation"])  # Convert to radians
    sx = params["scale_x"]
    sy = params["scale_y"]
    shear = np.radians(params["shear"])  # Convert to radians
    tx = params["tx"]
    ty = params["ty"]

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Correct scale matrix
    scale_matrix = np.array([[sx, 0], [0, sy]])

    # Correct shear matrix (applied in the correct order)
    shear_matrix = np.array([[1, np.tan(shear)], [0, 1]])

    # Correct rotation matrix
    rotation_matrix = np.array([[cos_theta, -sin_theta], 
                                [sin_theta, cos_theta]])

    # Compute final transformation matrix in the correct order: Scale → Shear → Rotation
    A = rotation_matrix @ shear_matrix @ scale_matrix

    # Construct full affine transformation matrix
    matrix = np.array([[A[0, 0], A[0, 1], tx], 
                       [A[1, 0], A[1, 1], ty], 
                       [0, 0, 1]])

    return matrix

def rescale_and_maskout(he_proc, he_mask):
    if he_mask is None:
        return he_proc

    desired_mask_size = max(he_proc.shape[0], he_proc.shape[1])

    new_size = (desired_mask_size, desired_mask_size)
    mask_resized = F.interpolate(he_mask.unsqueeze(0).unsqueeze(0),
                                 size=new_size,
                                 mode='nearest').squeeze(0).squeeze(0)
    # using nearest because this is a binary mask

    if he_proc.shape[0] < he_proc.shape[1]:
        mask_resized = mask_resized[:he_proc.shape[0], :]
    else:
        mask_resized = mask_resized[:, :he_proc.shape[1]]

    if len(he_proc.shape) == 3:
        if he_proc.shape[-1] == 3:  # HWC
            mask = einops.repeat(mask_resized, "H W -> H W C", C=3)
        elif he_proc.shape[0] == 3:  # CHW
            mask = einops.repeat(mask_resized, "H W -> C H W", C=3)
        else:
            assert False

        return mask * he_proc + (1 - mask) * torch.ones(he_proc.shape) * 255
    else:
        return mask_resized * he_proc

def downsample_cv2(img, scale_factor):
            """ Downsample using OpenCV while preserving axis scale. """
            h, w = img.shape[:2]
            img_small = cv2.resize(img, (w // scale_factor, h // scale_factor),
                                   interpolation=cv2.INTER_NEAREST)
            return img_small
class ManualBlockAligner:

    def __init__(self, cf):

        self.cf = cf

        # Block state
        self.align_results: Dict = None
        self.align_viz: Dict = None
        self.curr_block: pd.DataFrame = None
        self.curr_block_str: str = None
        self.flat_mask = None
        self.flat_mask_neg = None

        # Slide pair state
        self.he_i: int = None
        self.ihc_i: int = None

        self.he_thumbnail: np.array = None
        self.ihc_thumbnail: np.array = None
        self.he_proc: np.array = None
        self.ihc_proc: np.array = None

        self.he_thumbnail_orig: np.array = None
        self.ihc_thumbnail_orig: np.array = None
        self.he_proc_orig: np.array = None
        self.ihc_proc_orig: np.array = None

        self.latest_guess = None

    def init_one_block(self, block):

        with open(opj(self.cf["prev_exp"]["out_dir"], f"{block}_align.pkl"),
                  "rb") as fd:
            self.align_results = pickle.load(fd)
        with open(
                opj(self.cf["prev_exp"]["out_dir"], f"{block}_align_viz.pkl"),
                "rb") as fd:
            self.align_viz = pickle.load(fd)

        mask_annot = tifffile.imread(
            opj(self.cf["infra"]["annot_root"], "block_annot",
                f"{block}.tiff"))
        mask_annot_meta = pd.read_csv(
            opj(self.cf["infra"]["annot_root"], "meta",
                f"{block}_sections_annot_meta.csv"))

        section_mask_both = [get_sections_annot(m) for m in mask_annot]
        section_mask = [smb[0] for smb in section_mask_both]
        section_mask_neg = [smb[1] for smb in section_mask_both]

        self.flat_mask = list(chain.from_iterable(section_mask))
        self.flat_mask_neg = list(chain.from_iterable(section_mask_neg))
        lengths = list(map(len, section_mask))  # Lengths of each sublist
        starts = [0] + list(accumulate(
            lengths[:-1]))  # Start idx of each sublist
        list_indices = [
            list(range(start, start + length))
            for start, length in zip(starts, lengths)
        ]

        mask_annot_meta["sections_mask"] = list_indices
        curr_block = mask_annot_meta.explode("sections_mask").reset_index(
            drop=True)
        self.curr_block = curr_block[~(
            curr_block["comment"] == "RM")].reset_index(drop=True)
        #curr_block = curr_block.iloc[[0,3,5,6,7,-1]].reset_index(drop=True) # for debugging
        #curr_block = curr_block.iloc[[0,3]].reset_index(drop=True) # for debugging

        assert self.curr_block['UM Accession'].nunique() == 1
        assert self.curr_block['Block'].nunique() == 1

        self.curr_block_str = ".".join(
            (self.curr_block["UM Accession"][0], self.curr_block["Block"][0]))

        self.curr_he = self.curr_block[self.curr_block["Stain"] == "H&E"]

    def set_align_pair(self, he_i, ihc_i, do_viz=False):

        self.he_i = he_i
        self.ihc_i = ihc_i

        he_s = self.curr_he.iloc[he_i]
        he_slide = openslide.OpenSlide(
            opj(self.cf["infra"]["svs_root"], he_s["path"]))
        self.he_thumbnail_orig, he_scale_i = get_thumbnail(he_slide)
        src_is_he = (he_s["Stain"] == "H&E")
        self.he_proc_orig = basic_preprocessing(self.he_thumbnail_orig,
                                                red_only=src_is_he)

        self.he_thumbnail = rescale_and_maskout(
            self.he_thumbnail_orig, self.flat_mask[he_s["sections_mask"]])
        self.he_proc = rescale_and_maskout(
            self.he_proc_orig, self.flat_mask[he_s["sections_mask"]])

        ihc_s = self.curr_block.iloc[ihc_i]
        ihc_slide = openslide.OpenSlide(
            opj(self.cf["infra"]["svs_root"], ihc_s["path"]))
        self.ihc_thumbnail_orig, ihc_scale_j = get_thumbnail(ihc_slide)
        tgt_is_he = (ihc_s["Stain"] == "H&E")
        self.ihc_proc_orig = basic_preprocessing(self.ihc_thumbnail_orig,
                                                 red_only=tgt_is_he)

        print(f"{he_s['Stain']} {he_i} -> {ihc_s['Stain']} {ihc_i}")

        
        self.ihc_thumbnail = rescale_and_maskout(
            self.ihc_thumbnail_orig, self.flat_mask[ihc_s["sections_mask"]])
        self.ihc_proc = rescale_and_maskout(
            self.ihc_proc_orig, self.flat_mask[ihc_s["sections_mask"]])
        
        if do_viz:
            fig, ax = plt.subplots(1, 2, figsize=(20, 20))
            he_im_viz = np.array(self.he_thumbnail).astype(np.uint8)
            ihc_im_viz = np.array(self.ihc_thumbnail).astype(np.uint8)
            ax[0].imshow(downsample_cv2(he_im_viz, 2),
                     extent=[0, he_im_viz.shape[1], he_im_viz.shape[0], 0])
            ax[1].imshow(downsample_cv2(ihc_im_viz, 2),
                     extent=[0, ihc_im_viz.shape[1], ihc_im_viz.shape[0], 0])
            plt.show()

            fig, ax = plt.subplots(1, 2, figsize=(20, 20))
            he_mask_viz = np.array(self.he_proc*255).astype(np.uint8)
            ihc_mask_viz = np.array(self.ihc_proc*255).astype(np.uint8)
            ax[0].imshow(downsample_cv2(he_mask_viz,2), extent=[0, he_mask_viz.shape[1], he_mask_viz.shape[0], 0])
            ax[1].imshow(downsample_cv2(ihc_mask_viz,2),extent=[0, ihc_mask_viz.shape[1], ihc_mask_viz.shape[0], 0] )
            plt.show()

            previous_fit = decompose_affine_matrix(
                self.align_results["matrices"][he_i][ihc_i])
            self.visualize_transform(previous_fit)
            print(pd.DataFrame([previous_fit]).T)
    
    def chain_align_with_intermed(self, intermed_i):
        he_to_intermed = self.align_results["matrices"][
                self.he_i, intermed_i]
        intermed_to_ihc = self.align_results["matrices"][
                intermed_i, self.ihc_i]
        
        self.latest_guess = intermed_to_ihc @ he_to_intermed
        return self.latest_guess

    def generate_transform_viz(self, guess):
        transformed1 = cv2.warpAffine(
            np.array(self.he_proc).squeeze().astype(np.float64),
            guess[:2, ...].squeeze(),
            np.array(self.ihc_proc).squeeze().shape[::-1])
        b = np.zeros(transformed1.shape)
        stacked_im = np.dstack(
            (transformed1,
             np.array(self.ihc_proc_orig).squeeze().astype(np.float64), b))

        transformed2 = cv2.warpAffine(
            np.array(self.he_thumbnail).squeeze().astype(np.float64),
            guess[:2, ...].squeeze(),
            np.array(self.ihc_proc_orig).squeeze().shape[::-1])
        overlay = cv2.addWeighted(
            transformed2, .2,
            np.array(self.ihc_thumbnail).squeeze().astype(np.float64), .8,
            0).astype(np.uint8)
        return stacked_im, overlay

    def visualize_transform(self, guess):
        guess = reconstruct_affine_matrix(guess)

        self.latest_guess = guess
        stacked_im, overlay = self.generate_transform_viz(guess)
        fig, ax = plt.subplots(1, 2, figsize=(15, 20))
        ax[0].imshow(downsample_cv2(overlay, 4),
                     extent=[0, overlay.shape[1], overlay.shape[0], 0])
        ax[1].imshow(downsample_cv2(stacked_im, 4),
                     extent=[0, stacked_im.shape[1], stacked_im.shape[0], 0])
        plt.show()

    def commit_slide_pair(self, guess=None, do_viz=False):
        print(f"Commiting {(self.he_i, self.ihc_i)}")

        if guess is None:
            guess = self.latest_guess
        else:
            guess = reconstruct_affine_matrix(guess)

        stacked_im, overlay = self.generate_transform_viz(guess)

        self.align_results["matrices"][self.he_i, self.ihc_i] = guess
        self.align_results["num_matches"][self.he_i, self.ihc_i] = 0

        self.align_viz["align_viz_binary"][self.he_i][self.ihc_i] = stacked_im
        self.align_viz["align_viz_im"][self.he_i][self.ihc_i] = overlay

        print("OK")
    
        if do_viz:
            self.visualize_transform(decompose_affine_matrix(guess))

    def commit_save_block(self, do_save=True):

        if do_save:
            os.makedirs(self.cf["manual"]["out_dir"], exist_ok=True)
            with open(
                    opj(self.cf["manual"]["out_dir"],
                        f"{self.curr_block_str}_align.pkl"), "wb") as fd:
                pickle.dump(self.align_results, fd)

            with open(
                    opj(self.cf["manual"]["out_dir"],
                        f"{self.curr_block_str}_align_viz.pkl"), "wb") as fd:
                pickle.dump(self.align_viz, fd)
        else:
            print("NO SAVE")

        plt.rcParams['pdf.fonttype'] = 42

        fig, axs = plt.subplots(len(self.align_viz["align_viz_binary"]) + 1,
                                len(self.align_viz["align_viz_binary"][0]) + 1,
                                figsize=(20, 5),
                                dpi=300)
        for i, ims_i in enumerate(self.align_viz["align_viz_im"]):
            for j, im_ij in enumerate(ims_i):
                axs[i + 1][j + 1].imshow(im_ij)
                axs[i + 1][j + 1].axis("off")

        for im_i, (i, r) in zip(self.align_viz["he_im"],
                                self.curr_he.iterrows()):
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
        for im_i, (i,
                   r) in zip(self.align_viz["all_im"][:len(self.curr_block)],
                             self.curr_block.iterrows()):
            axs[0][i + 1].imshow(im_i.astype(np.uint8))
            axs[0][i + 1].axis("off")
            axs[0][i + 1].set_title(r["Stain"])

        axs[0][0].axis("off")

        plt.tight_layout()
        if do_save:
            plt.savefig(
                opj(self.cf["manual"]["out_dir"],
                    f"{self.curr_block_str}_im_align.pdf"))

        fig, axs = plt.subplots(len(self.align_viz["align_viz_binary"]) + 1,
                                len(self.align_viz["align_viz_binary"][0]) + 1,
                                figsize=(20, 5),
                                dpi=300)
        for i, ims_i in enumerate(self.align_viz["align_viz_binary"]):
            for j, im_ij in enumerate(ims_i):
                axs[i + 1][j + 1].imshow(im_ij)
                axs[i + 1][j + 1].axis("off")

        for im_i, (i, r) in zip(self.align_viz["he_binary"],
                                self.curr_he.iterrows()):
            im = (einops.repeat(im_i, "h w -> h w c", c=3) * 255).astype(
                np.uint8)
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

        for im_i, (i, r) in zip(
                self.align_viz["all_binary"][:len(self.curr_block)],
                self.curr_block.iterrows()):
            im = (einops.repeat(im_i, "h w -> h w c", c=3) * 255).astype(
                np.uint8)
            im[:, :, [0, 2]] = 0
            axs[0][i + 1].imshow(im)
            axs[0][i + 1].axis("off")
            axs[0][i + 1].set_title(r["Stain"])

        axs[0][0].axis("off")

        plt.tight_layout()
        if do_save:
            plt.savefig(
                opj(self.cf["manual"]["out_dir"],
                    f"{self.curr_block_str}_mask_align.pdf"))
            print("OK")
