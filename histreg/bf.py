import openslide
import matplotlib.pyplot as plt
import numpy as np
import glob
import json
import pandas as pd
from tqdm import tqdm
from typing import List
import logging
import cv2
from PIL import Image, ImageOps
from skimage.morphology import skeletonize
from skimage import measure
from scipy.spatial import distance
from sklearn.decomposition import PCA

import torch
import torch.nn.functional as F


def create_weight_map(shape, method='gaussian', sigma=50):
    rows, cols = shape
    if method == 'gaussian':
        # Create a Gaussian kernel centered in the middle of the image
        center_x, center_y = cols // 2, rows // 2
        x, y = np.meshgrid(np.linspace(0, cols - 1, cols),
                           np.linspace(0, rows - 1, rows))
        weight_map = np.exp(-((x - center_x)**2 + (y - center_y)**2) /
                            (2 * sigma**2))
    elif method == 'linear':
        # Linear decay from center
        center_x, center_y = cols // 2, rows // 2
        x, y = np.meshgrid(np.linspace(0, cols - 1, cols),
                           np.linspace(0, rows - 1, rows))
        weight_map = 1 - (np.abs(x - center_x) +
                          np.abs(y - center_y)) / (rows + cols)
        weight_map = np.clip(weight_map, 0, 1)
    else:
        # Default to uniform weighting if no valid method specified
        weight_map = np.ones((rows, cols), dtype=np.float32)

    return weight_map


def apply_transformation(mask1, shape2, H):
    warped_img1 = cv2.warpPerspective(mask1, H, shape2[::-1], borderValue=0)
    return warped_img1


def get_homography_matrx_(tx, ty, scale, angle, shear):

    cos_a = np.cos(np.radians(angle)) * scale
    sin_a = np.sin(np.radians(angle)) * scale

    return np.array([[cos_a, sin_a + shear, tx], [-sin_a + shear, cos_a, ty],
                     [0, 0, 1]])


def get_homography_matrx(im_size, tx, ty, scale, angle, shear):
    """
    Compute the homography matrix for affine transformation centered around the image center.
    
    Parameters:
        im_size (tuple): (height, width) of the image.
        tx (float): Translation in x direction.
        ty (float): Translation in y direction.
        scale (float): Scaling factor.
        angle (float): Rotation angle in degrees.
        shear (float): Shear angle in degrees.
    
    Returns:
        np.ndarray: 3x3 homography matrix.
    """
    h, w = im_size
    cx, cy = w / 2, h / 2  # Compute image center

    # Convert angles to radians
    angle_rad = np.radians(angle)
    shear_rad = np.radians(shear)

    # Translation matrix
    T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])

    # Scaling matrix
    S = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])

    # Rotation matrix
    R = np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                  [np.sin(angle_rad), np.cos(angle_rad), 0], [0, 0, 1]])

    # Shear matrix
    Sh = np.array([[1, np.tan(shear_rad), 0], [0, 1, 0], [0, 0, 1]])

    # Translate to center, apply transformations, then translate back
    T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])

    T2 = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]])

    # Compute final homography matrix
    H = T2 @ T @ R @ Sh @ S @ T1

    return H


def binarize_otsu(image):
    _, binary = cv2.threshold(image, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def get_pca_transform(mask):
    coords = np.column_stack(np.where(mask > 0))
    pca = PCA(n_components=2)
    pca.fit(coords)
    center = pca.mean_
    angle = np.arctan2(pca.components_[0, 1], pca.components_[0,
                                                              0]) * 180 / np.pi
    return center, angle


def do_bf(mask1, mask2, x_range, y_range, d_range, scale_range, shear_range):

    best_iou = 0
    best_params = [0, 0, 0, 0, 0]

    # Define parameter ranges for brute force
    for tx in tqdm(x_range):
        for ty in y_range:
            for scale in scale_range:
                for angle in d_range:
                    for shear in shear_range:
                        H = get_homography_matrx(mask1.shape, tx, ty, scale,
                                                 angle, shear)
                        transformed1 = apply_transformation(
                            mask1, mask2.shape, H)

                        visualization = np.zeros(
                            (mask2.shape[0], mask2.shape[1], 3),
                            dtype=np.uint8)
                        visualization[..., 0] = transformed1
                        visualization[..., 1] = mask2

                        # Save the visualization
                        #cv2.imwrite(f"a{angle}.png", visualization)

                        current_iou = (((mask2 / 255) *
                                        (transformed1 / 255) * 2)).sum() / (
                                            (mask2 / 255).sum() +
                                            (transformed1 / 255).sum())
                        if current_iou > best_iou:
                            best_iou = current_iou
                            best_params = (tx, ty, scale, angle, shear)
                            logging.info(f"+{current_iou}")
                            logging.info(f"+{best_params}")

    return best_params, best_iou


def match_bf(mask1, mask2):

    mask1 = binarize_otsu((mask1.numpy() * 255).astype(np.uint8))
    mask2 = binarize_otsu((mask2.numpy() * 255).astype(np.uint8))

    c1, angle1 = get_pca_transform(mask1)
    c2, angle2 = get_pca_transform(mask2)

    coarse_x_range = np.arange(-30, 31, 5, dtype=float) + (c2[1] - c1[1])  # 13
    coarse_y_range = np.arange(-30, 31, 5, dtype=float) + (c2[0] - c1[0])  # 13
    coarse_d_range = np.arange(-180, 180, 10, dtype=float)  # 18
    coarse_scale_range = [1] #np.arange(0.9, 1.11, 0.1, dtype=float)  # 5
    coarse_shear_range = [0]

    coarse_opt, coarse_iou = do_bf(mask1,
                          mask2,
                          x_range=coarse_x_range,
                          y_range=coarse_y_range,
                          d_range=coarse_d_range,
                          scale_range=coarse_scale_range,
                          shear_range=coarse_shear_range)

    #fine_x_range = np.arange(-4, 5, 1, dtype=float) + coarse_opt[0]  # 9
    #fine_y_range = np.arange(-4, 5, 1, dtype=float) + coarse_opt[1]  # 9
    #fine_scale_range = np.arange(-0.05, 0.06, 0.05,
    #                             dtype=float) + coarse_opt[2]  # 3
    #fine_d_range = np.arange(-19, 20, 1, dtype=float) + coarse_opt[3]  # 40
    #fine_shear_range = np.array([0]) + coarse_opt[4]

    #fine_opt, fine_iou = do_bf(mask1,
    #                           mask2,
    #                           x_range=fine_x_range,
    #                           y_range=fine_y_range,
    #                           d_range=fine_d_range,
    #                           scale_range=fine_scale_range,
    #                           shear_range=fine_shear_range)

    return get_homography_matrx(mask1.shape, *coarse_opt), coarse_iou
