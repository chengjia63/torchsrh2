import logging
import torch
import torchvision.transforms.functional as TF
import numpy as np
import cv2
import openslide
import pandas as pd
import pickle
from madeleine.hest_modules.wsi import OpenSlideWSI
import tifffile
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from torch.nn import functional as F
import einops
import math
from tqdm import tqdm
import os
from os.path import join as opj
import re
from itertools import chain, accumulate, compress
from glob import glob

import geopandas as gpd
from shapely.affinity import affine_transform

from shapely.ops import unary_union

import pyvips
import pickle


def sanitize_string(string):
    return re.sub(r'[^a-zA-Z0-9]', '', string)


def get_sections_annot(he_annot):
    annot_mask = np.all(he_annot == np.array([[[0, 255, 0]]]), axis=2)
    contours, _ = cv2.findContours(annot_mask.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return [None]

    # Create a filled mask for each contour
    masked_ims = []
    for i, contour in enumerate(contours):
        # Create an empty mask
        mask = np.zeros_like(annot_mask, dtype=float)
        # Fill the contour
        cv2.drawContours(mask, [contour], -1, 1, thickness=cv2.FILLED)
        masked_ims.append(torch.tensor(mask))

    return masked_ims


def get_sections_annot2(he_annot, tissue_mask, wsi_size):
    annot_mask = np.all(he_annot == np.array([[[0, 255, 0]]]), axis=2)
    contours, _ = cv2.findContours(annot_mask.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(contours)
    if len(contours) == 0:
        return [tissue_mask]

    scale_factor = max(wsi_size) / he_annot.shape[0]
    contours = [c * scale_factor for c in contours]
    contour_polygons = [Polygon(c.reshape(-1, 2)) for c in contours]
    new_masks = [
        tissue_mask[tissue_mask['geometry'].apply(
            lambda x: True if cp.intersects(x) else False)]
        for cp in contour_polygons
    ]

    return new_masks


def save_as_dzi(np_array, output_dir, tile_size=256, overlap=0, layout='dz'):
    """
    Saves a large numpy array as a Deep Zoom Image (DZI) for OpenSeadragon.

    Args:
        np_array (numpy.ndarray): The input whole-slide image as a numpy array.
        output_dir (str): The directory to save the DZI output.
        tile_size (int): Size of each tile (default: 256).
        overlap (int): Tile overlap in pixels (default: 0).
        layout (str): The pyramid layout ('dz' for Deep Zoom, 'zoomify' for Zoomify).
    """
    # Convert the numpy array to pyvips image
    height, width, bands = np_array.shape
    if np_array.dtype != np.uint8:
        raise ValueError("Input numpy array must have dtype=np.uint8")

    # Flatten numpy array for pyvips (row-major order)
    vips_image = pyvips.Image.new_from_memory(np_array.tobytes(),
                                              width,
                                              height,
                                              bands,
                                              format='uchar')

    # Save as DZI
    vips_image.dzsave(output_dir,
                      tile_size=tile_size,
                      overlap=overlap,
                      layout=layout)
    print(f"DZI saved successfully at: {output_dir}")


def scale_alignment_matrix(x, scale_down_factor=64, scale_up_factor=64):

    scale_down = np.array([[1 / scale_down_factor, 0, 0],
                           [0, 1 / scale_down_factor, 0], [0, 0, 1]])

    scale_up = np.array([[scale_up_factor, 0, 0], [0, scale_up_factor, 0],
                         [0, 0, 1]])
    joint = scale_up @ x @ scale_down

    return joint


def wrap_image(src_slide,
               alignment_matrix,
               x0y0,
               wh=(8192, 8192),
               scale_down_factor=64,
               scale_up_factor=64):
    joint = scale_alignment_matrix(alignment_matrix, scale_down_factor,
                                   scale_up_factor)

    x_tgt, y_tgt = np.meshgrid(np.arange(x0y0[0], x0y0[0] + wh[0]),
                               np.arange(x0y0[1], x0y0[1] + wh[1]),
                               indexing="ij")

    # Homogeneous coordinates for the output tile
    ones = np.ones_like(x_tgt)
    coords_tgt = np.stack((x_tgt, y_tgt, ones), axis=-1)  # Shape: (H, W, 3)

    # Transform to source coordinates
    coords_src = (coords_tgt @ joint.T)[..., :2]  # Shape: (H, W, 2)

    min_x_src = int(np.floor(np.min(coords_src[..., 0])))
    max_x_src = int(np.ceil(np.max(coords_src[..., 0])))
    min_y_src = int(np.floor(np.min(coords_src[..., 1])))
    max_y_src = int(np.ceil(np.max(coords_src[..., 1])))

    if ((min_x_src < 0) or (min_y_src < 0) or (max_x_src < 0)
            or (max_y_src < 0) or (min_x_src >= src_slide.dimensions[0])
            or (min_y_src >= src_slide.dimensions[1])
            or (max_x_src >= src_slide.dimensions[0])
            or (max_y_src >= src_slide.dimensions[1])):
        #print((x0y0), (min_x_src, min_y_src, max_x_src, max_y_src))
        return x0y0, np.ones((wh[0], wh[1], 3), dtype=np.uint8) * 255

    # get src im
    he1 = np.array(
        src_slide.read_region(
            (min_x_src, min_y_src), 0,
            (max_x_src - min_x_src, max_y_src - min_y_src)).convert("RGB"))

    coords_src[..., 0] -= min_x_src
    coords_src[..., 1] -= min_y_src

    # Normalize source coordinates to [-1, 1] for PyTorch grid_sample
    h_src, w_src = he1.shape[:2]
    coords_src[..., 0] = 2.0 * coords_src[..., 0] / (w_src - 1) - 1.0
    coords_src[..., 1] = 2.0 * coords_src[..., 1] / (h_src - 1) - 1.0

    # Convert cropped region to PyTorch tensor and move to GPU
    cropped_tensor = einops.rearrange(
        torch.from_numpy(he1), "h w c -> c h w").unsqueeze(0).float()  #.cuda()

    # Reshape source coordinates for grid_sample
    grid = torch.from_numpy(coords_src).unsqueeze(
        0).float()  #.cuda()  # Shape: (1, H, W, 2)

    # Perform sampling
    output_tile = torch.nn.functional.grid_sample(cropped_tensor,
                                                  grid,
                                                  mode="bilinear",
                                                  align_corners=False)

    return None, einops.rearrange(
        output_tile.detach().cpu().squeeze().to(torch.uint8),
        "c h w -> w h c").numpy()  # to debug


def make_warp_status_thumbnail(patch_cords, fail_coords):
    thumbnail_size = (patch_cords.max(axis=0) // 512) + 1
    all_pixels = np.zeros([thumbnail_size[1], thumbnail_size[0], 3],
                          dtype=np.uint8)

    for b in patch_cords:
        all_pixels[b[1] // 512, b[0] // 512] = [0, 255, 0]

    if fail_coords:
        for b in fail_coords:
            all_pixels[b[1] // 512, b[0] // 512] = [255, 0, 0]

    return all_pixels


def get_seg_out(segmentation_root, x):
    candidates = glob(
        opj(segmentation_root, "*", "segmentation/pkl",
            f"{x}_tissue_mask.pkl"))
    assert len(candidates) == 1
    return candidates[0]


def load_tissue_masks(fname):
    with open(fname, "rb") as fd:
        tissue_mask = pickle.load(fd)
    return tissue_mask


def patch_one_block(su_b_df,
                    sections_annot_root,
                    alignment_results_dir,
                    segmentation_root,
                    svs_root,
                    out_dir,
                    patch_size=512,
                    out_size=256):

    su_b = su_b_df["block"]
    logging.info(f"Block {su_b}")

    logging.info(f"Block {su_b} -- loading materials")
    mask_annot = tifffile.imread(
        opj(sections_annot_root, f"block_annot/{su_b}.tiff"))
    mask_annot_meta = pd.read_csv(
        opj(sections_annot_root, f"meta/{su_b}_sections_annot_meta.csv"))
    #mask_annot_meta = mask_annot_meta.rename(
    #    columns={"Unnamed: 0": "main_idx"})
    
    if su_b_df["which"] == "one_section_only":
        pass
    else:
        with open(
                opj(alignment_results_dir, su_b_df["which"], f"{su_b}_align.pkl"),
                "rb") as fd:
            alignment_result = pickle.load(fd)

    he_svs_fname = [
        x.split("/")[-1].removesuffix(".svs")
        for x in mask_annot_meta.loc[mask_annot_meta["Stain"] == "H&E",
                                     "path"].tolist()
    ]

    logging.info(f"Block {su_b} -- parsing / merging HE segmentations")
    he_patch_dir = [get_seg_out(segmentation_root, x) for x in he_svs_fname]
    tissue_mask = [load_tissue_masks(x) for x in he_patch_dir]
    section_mask = [get_sections_annot(m) for m in mask_annot]

    to_remove = (mask_annot_meta["comment"] == "RM")
    mask_annot_meta=mask_annot_meta[~to_remove].reset_index(              drop=True) 
    section_mask = list(compress(section_mask, ~to_remove))

    flat_mask = list(chain.from_iterable(section_mask))
    lengths = list(map(len, section_mask))  # Lengths of each sublist
    starts = [0] + list(accumulate(
        lengths[:-1]))  # Start indices of each sublist in the flattened list
    list_indices = [
        list(range(start, start + length))
        for start, length in zip(starts, lengths)
    ]

    mask_annot_meta["sections_mask"] = list_indices


    all_slides = [
        openslide.OpenSlide(opj(svs_root, r["path"]))
        for i, r in mask_annot_meta.iterrows()
    ]

    he_df = mask_annot_meta[mask_annot_meta["Stain"] == "H&E"]
    section_tissue_masks0 = [
        get_sections_annot2(mask_annot[0], tissue_mask[0],
                            all_slides[0].dimensions)
        for i, _ in he_df.iterrows()
    ]

    flatten_he_masks = list(chain(*section_tissue_masks0))
    flatten_he_idx = list(chain(*he_df["sections_mask"].tolist()))

    merged_he_mask = flatten_he_masks[0]
    if len(flatten_he_masks) > 1:
        for i, m in zip(flatten_he_idx[1:], flatten_he_masks[1:]):
            joint = scale_alignment_matrix(
                alignment_result["matrices"][flatten_he_idx[i], flatten_he_idx[0]],
                scale_down_factor=64,
                scale_up_factor=64)
            (a, b, xoff), (d, e, yoff) = joint[0], joint[1]

            #m["geometry"] = m["geometry"].apply(lambda geom: affine_transform(geom, [a, b, d, e, xoff, yoff]))

            union_geom = unary_union(
                merged_he_mask.geometry.tolist() +
                m["geometry"].apply(lambda geom: affine_transform(
                    geom, [a, b, d, e, xoff, yoff])).tolist())
            merged_he_mask = gpd.GeoDataFrame(geometry=[union_geom],
                                              crs=merged_he_mask.crs)

    curr_block = mask_annot_meta.explode("sections_mask").reset_index(
        drop=False).rename({"index": "svs_idx"}, axis=1)
    curr_block.to_csv(opj(out_dir, f"{su_b}_blockmeta.csv"), index=False)

    logging.info(f"Block {su_b} -- getting patch coordinates")
    patcher = OpenSlideWSI(img=all_slides[0]).create_patcher(
        patch_size=patch_size,
        src_pixel_size=1,
        dst_pixel_size=1,
        overlap=0,
        mask=merged_he_mask,
        coords_only=True)
    patch_cords = patcher.valid_coords
    np.save(opj(out_dir, f"{su_b}_coords_{out_size}.npy"), patch_cords)
    #patch_cords = np.load(opj(out_dir, f"{su_b}_coords_{out_size}.npy"))

    #WE ARE DOING ONE MEMMAP FILE FOR THE ENTIRE BLOCK
    logging.info(f"Block {su_b} -- start saving")
    tensor_shape = (len(patch_cords), len(curr_block), out_size, out_size,
                    3)
    dtype = 'uint8'
    fd_full = np.memmap(opj(out_dir, f"{su_b}_patches_{out_size}.dat"),
                        dtype='uint8',
                        mode='w+',
                        shape=tensor_shape)
    element_size = np.dtype(dtype).itemsize
    del fd_full

    fail_coords = [[] for _ in range(len(curr_block))]

    for j, xy in tqdm(enumerate(patch_cords), total=len(patch_cords)):

        he0 = np.array(all_slides[curr_block.iloc[0]["svs_idx"]].read_region(
            tuple(xy), 0, (patch_size, patch_size)).convert("RGB"))
        he0 = cv2.resize(he0, (out_size, out_size), interpolation=cv2.INTER_AREA)
        ims_xy = [he0]

        for (i, r) in curr_block[1:].iterrows():
            fail_coord, he1 = wrap_image(all_slides[r["svs_idx"]],
                                         alignment_result["matrices"][0, i],
                                         tuple(xy), (patch_size, patch_size))
            he1 = cv2.resize(he1, (out_size, out_size), interpolation=cv2.INTER_AREA)
            if fail_coord:
                fail_coords[i].append(fail_coord)
            ims_xy.append(he1)

        # Calculate the offset (in bytes) where this chunk should be written
        offset = j * len(
            curr_block) * out_size * out_size * 3 * element_size

        # Open a new memmap for just this chunk and write
        data_memmap = np.memmap(opj(out_dir,
                                    f"{su_b}_patches_{out_size}.dat"),
                                dtype=dtype,
                                mode='r+',
                                shape=(len(curr_block), out_size, out_size,
                                       3),
                                offset=offset)
        data_memmap[:] = np.stack(ims_xy)
        data_memmap.flush()

        del data_memmap
        del ims_xy

    with open(opj(out_dir, f"{su_b}_coords_failed_{out_size}.pkl"),
              "wb") as fd:
        pickle.dump(fail_coords, fd)


    fd_full = np.memmap(opj(out_dir, f"{su_b}_patches_{out_size}.dat"), dtype='uint8', mode='r', shape=(len(patch_cords), len(curr_block), 256, 256, 3))
    patch_mean = np.stack([fdp.mean(axis=(1,2)) for fdp in fd_full])
    pixel_means = patch_mean.mean(axis=0)
    curr_block["pixel_mean_r"] = pixel_means[:,0].tolist()
    curr_block["pixel_mean_g"] = pixel_means[:,1].tolist()
    curr_block["pixel_mean_b"] = pixel_means[:,2].tolist()

    curr_block.to_csv(opj(out_dir, f"{su_b}_blockmeta.csv"), index=False)

    logging.info(f"Block {su_b} -- OK")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format=
        "[%(levelname)-s|%(asctime)s|%(filename)s:%(lineno)d|%(funcName)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler()])
    logging.info("Block patching log")

    accepted_blocks = pd.read_csv(
        "/nfs/turbo/umms-tocho/code/chengjia/torchsrh2/histreg/out/accepted.csv")
    #accepted_blocks = pd.read_csv(
    #    "/nfs/turbo/umms-tocho/code/chengjia/torchsrh2/histreg/out/one_section.csv")

    env_var = dict(os.environ)
    if "SLURM_ARRAY_TASK_ID" in env_var:
        taskid = int(env_var["SLURM_ARRAY_TASK_ID"])
    else:
        taskid = 0


    already_patched = glob("/nfs/turbo/umms-tocho-snr/exp/chengjia/mns_block_patch/*_coords_failed_256.pkl")
    already_patched = set([p.split("/")[-1].removesuffix("_coords_failed_256.pkl") for p in already_patched])
    #to_patch = {"SU-15-62441.A3"}
    accepted_blocks = accepted_blocks[~ accepted_blocks["block"].isin(already_patched)]

    logging.info(f"GOT TASK ID {taskid}")
    patch_one_block(
        su_b_df=accepted_blocks.iloc[taskid],
        patch_size=512,
        out_size=256,
        sections_annot_root=
        "/nfs/turbo/umms-tocho/code/chengjia/torchsrh2/ts2/playgrounds/pixel_alignment/sections_annot2",
        alignment_results_dir="/nfs/turbo/umms-tocho-snr/exp/chengjia/",
        segmentation_root=
        "/nfs/turbo/umms-tocho-snr/exp/chengjia/madeleine_patch",
        svs_root="/nfs/mm-isilon/brainscans/dropbox/Slide_Incoming/svs",
        out_dir="/nfs/turbo/umms-tocho-snr/exp/chengjia/mns_block_patch")
        #out_dir=".")
