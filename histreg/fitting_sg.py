"""Fit transformations using superpoint and superglue.

Code adapted from:
https://github.com/MWod/DeeperHistReg/
https://github.com/magicleap/SuperGluePretrainedNetwork
License from these repository applies.

"""
from typing import Dict, Iterable, Union, List, Tuple
from copy import deepcopy
import logging

import numpy as np
import cv2

import torch
from torch import nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms

import einops

log = logging.getLogger("histreg.fitting_sg")

default_match_cf = {
    "superglue_params": {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': 3000
        },
        'superglue': {
            'weights': 'outdoor',
            'sinkhorn_iterations': 30,
            'match_threshold': 0.3,
        }
    },
    'superpoint_weights_path': "models/superpoint_v1.pth",
    'superglue_weights_path': "models/superglue_outdoor.pth",
    'transform_type': "affine",
    'device': "cuda"
}


def np_to_tensor(x: np.array):
    """
    Converts a NumPy array representing an image into a PyTorch tensor 
    with channel-first format (C, H, W) and ensures it is contiguous in memory.

    Args:
        x (np.array): A NumPy array with shape (H, W, C), where:
                      - H is the height of the image.
                      - W is the width of the image.
                      - C is the number of channels (e.g., 3 for RGB images).

    Returns:
        torch.Tensor: A PyTorch tensor with shape (C, H, W), converted to float
                      and made contiguous in memory.
    """
    return torch.tensor(einops.rearrange(
        x, "h w c -> c h w")).contiguous().float()


def normalize(x: torch.Tensor) -> torch.Tensor:
    """
    Normalizes a PyTorch tensor image channel-wise.

    Args:
        x (torch.Tensor): A PyTorch tensor representing an image with shape (C, H, W),
            where C is the number of channels, H is the height, and W is
            the width.

    Returns:
        torch.Tensor: A normalized PyTorch tensor with the same shape as the input,
                    where each channel is normalized to have values in the range [0, 1].
    """
    max_ = einops.reduce(x, 'c h w -> c', 'max')
    min_ = einops.reduce(x, 'c h w -> c', 'min')
    return torchvision.transforms.functional.normalize(x, min_, max_ - min_)


def convert_to_gray(x: torch.Tensor) -> torch.Tensor:
    """
    Converts an RGB image tensor to a grayscale image tensor.

    Args:
        x (torch.Tensor): A PyTorch tensor representing an RGB image with shape 
            (C, H, W), where C = 3 (number of color channels), H is the height, 
            and W is the width.

    Returns:
        torch.Tensor: A PyTorch tensor representing a grayscale image with shape 
            (1, H, W), where the single channel represents the grayscale intensity.
    """
    return torchvision.transforms.functional.rgb_to_grayscale(
        x, num_output_channels=1)


def basic_preprocessing(x: np.ndarray) -> torch.Tensor:
    """
    Performs preprocessing on an input image.

    Args:
        x (np.ndarray): A NumPy array representing the input image, typically 
            with pixel values normalized to the range [0, 1].

    Returns:
        torch.Tensor: A PyTorch tensor representing the preprocessed image with 
            pixel values normalized to the range [0, 1].

    Processing steps:
        1. Normalize the input image.
        2. Convert the image to grayscale and invert it.
        3. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for 
           contrast enhancement.
        4. Convert the processed image back to a PyTorch tensor.
    """
    x = normalize(x)
    x = 1 - convert_to_gray(x).squeeze()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    x = clahe.apply((np.array(x) * 255).astype(np.uint8))
    return torch.from_numpy((x.astype(np.float32) / 255))


def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert (nms_radius >= 0)

    def max_pool(x):
        return torch.nn.functional.max_pool2d(x,
                                              kernel_size=nms_radius * 2 + 1,
                                              stride=1,
                                              padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def remove_borders(keypoints, scores, border: int, height: int, width: int):
    """ Removes keypoints too close to the border """
    mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0]
                                            < (height - border))
    mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]


def top_k_keypoints(keypoints, scores, k: int):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0)
    return keypoints[indices], scores


def sample_descriptors(keypoints, descriptors, s: int = 8):
    """ Interpolate descriptors at keypoint locations """
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor([(w * s - s / 2 - 0.5),
                               (h * s - s / 2 - 0.5)], ).to(keypoints)[None]
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    args = {'align_corners': True} if torch.__version__ >= '1.3' else {}
    descriptors = torch.nn.functional.grid_sample(descriptors,
                                                  keypoints.view(b, 1, -1, 2),
                                                  mode='bilinear',
                                                  **args)
    descriptors = torch.nn.functional.normalize(descriptors.reshape(b, c, -1),
                                                p=2,
                                                dim=1)
    return descriptors


class SuperPoint(nn.Module):
    """SuperPoint Convolutional Detector and Descriptor
    SuperPoint: Self-Supervised Interest Point Detection and
    Description. Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. In CVPRW, 2019. https://arxiv.org/abs/1712.07629
    """
    default_config = {
        'descriptor_dim': 256,
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1,
        'remove_borders': 4,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(c5,
                                self.config['descriptor_dim'],
                                kernel_size=1,
                                stride=1,
                                padding=0)

        # path = Path(__file__).parent / 'weights/superpoint_v1.pth'
        # self.load_state_dict(torch.load(str(path)))

        mk = self.config['max_keypoints']
        if mk == 0 or mk < -1:
            raise ValueError('\"max_keypoints\" must be positive or \"-1\"')

        # print('Loaded SuperPoint model')

    def forward(self, data):
        """ Compute keypoints, scores, descriptors for image """
        # Shared Encoder
        x = self.relu(self.conv1a(data['image']))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Compute the dense keypoint scores
        cPa = self.relu(self.convPa(x))
        scores = self.convPb(cPa)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)
        scores = simple_nms(scores, self.config['nms_radius'])

        # Extract keypoints
        keypoints = [
            torch.nonzero(s > self.config['keypoint_threshold'])
            for s in scores
        ]
        scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]

        # Discard keypoints near the image borders
        keypoints, scores = list(
            zip(*[
                remove_borders(k, s, self.config['remove_borders'], h * 8, w *
                               8) for k, s in zip(keypoints, scores)
            ]))

        # Keep the k keypoints with highest score
        if self.config['max_keypoints'] >= 0:
            keypoints, scores = list(
                zip(*[
                    top_k_keypoints(k, s, self.config['max_keypoints'])
                    for k, s in zip(keypoints, scores)
                ]))

        # Convert (h, w) to (x, y)
        keypoints = [torch.flip(k, [1]).float() for k in keypoints]

        # Compute the dense descriptors
        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa)
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)

        # Extract descriptors
        descriptors = [
            sample_descriptors(k[None], d[None], 8)[0]
            for k, d in zip(keypoints, descriptors)
        ]

        return {
            'keypoints': keypoints,
            'scores': scores,
            'descriptors': descriptors,
        }


def MLP(channels: List[int], do_bn: bool = True) -> nn.Module:
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def normalize_keypoints(kpts, image_shape):
    """ Normalize keypoints locations based on image image_shape"""
    _, _, height, width = image_shape
    one = kpts.new_tensor(1)
    size = torch.stack([one * width, one * height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]


class KeypointEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""

    def __init__(self, feature_dim: int, layers: List[int]) -> None:
        super().__init__()
        self.encoder = MLP([3] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores):
        inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]
        return self.encoder(torch.cat(inputs, dim=1))


def attention(query: torch.Tensor, key: torch.Tensor,
              value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """

    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor) -> torch.Tensor:
        batch_dim = query.size(0)
        query, key, value = [
            l(x).view(batch_dim, self.dim, self.num_heads, -1)
            for l, x in zip(self.proj, (query, key, value))
        ]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim,
                                              self.dim * self.num_heads, -1))


class AttentionalPropagation(nn.Module):

    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim * 2, feature_dim * 2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):

    def __init__(self, feature_dim: int, layer_names: List[str]) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names))
        ])
        self.names = layer_names

    def forward(self, desc0: torch.Tensor,
                desc1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for layer, name in zip(self.layers, self.names):
            if name == 'cross':
                src0, src1 = desc1, desc0
            else:  # if name == 'self':
                src0, src1 = desc0, desc1
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1


def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor,
                            log_nu: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores: torch.Tensor, alpha: torch.Tensor,
                          iters: int) -> torch.Tensor:
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m * one).to(scores), (n * one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat(
        [torch.cat([scores, bins0], -1),
         torch.cat([bins1, alpha], -1)], 1)

    norm = -(ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


class SuperGlue(nn.Module):
    """SuperGlue feature matching middle-end
    Given two sets of keypoints and locations, we determine the
    correspondences by:
      1. Keypoint Encoding (normalization + visual feature and location fusion)
      2. Graph Neural Network with multiple self and cross-attention layers
      3. Final projection layer
      4. Optimal Transport Layer (a differentiable Hungarian matching algorithm)
      5. Thresholding matrix based on mutual exclusivity and a match_threshold
    The correspondence ids use -1 to indicate non-matching points.
    Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. SuperGlue: Learning Feature Matching with Graph Neural
    Networks. In CVPR, 2020. https://arxiv.org/abs/1911.11763
    """
    default_config = {
        'descriptor_dim': 256,
        'weights': 'indoor',
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 100,
        'match_threshold': 0.2,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.kenc = KeypointEncoder(self.config['descriptor_dim'],
                                    self.config['keypoint_encoder'])

        self.gnn = AttentionalGNN(feature_dim=self.config['descriptor_dim'],
                                  layer_names=self.config['GNN_layers'])

        self.final_proj = nn.Conv1d(self.config['descriptor_dim'],
                                    self.config['descriptor_dim'],
                                    kernel_size=1,
                                    bias=True)

        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

        # assert self.config['weights'] in ['indoor', 'outdoor']
        # path = Path(__file__).parent
        # path = path / 'weights/superglue_{}.pth'.format(self.config['weights'])
        # self.load_state_dict(torch.load(str(path)))
        # print('Loaded SuperGlue model (\"{}\" weights)'.format(
        #     self.config['weights']))

    def forward(self, data):
        """Run SuperGlue on a pair of keypoints and descriptors"""
        desc0, desc1 = data['descriptors0'], data['descriptors1']
        kpts0, kpts1 = data['keypoints0'], data['keypoints1']

        if kpts0.shape[1] == 0 or kpts1.shape[1] == 0:  # no keypoints
            shape0, shape1 = kpts0.shape[:-1], kpts1.shape[:-1]
            return {
                'matches0': kpts0.new_full(shape0, -1, dtype=torch.int),
                'matches1': kpts1.new_full(shape1, -1, dtype=torch.int),
                'matching_scores0': kpts0.new_zeros(shape0),
                'matching_scores1': kpts1.new_zeros(shape1),
            }

        # Keypoint normalization.
        kpts0 = normalize_keypoints(kpts0, data['image0'].shape)
        kpts1 = normalize_keypoints(kpts1, data['image1'].shape)

        # Keypoint MLP encoder.
        desc0 = desc0 + self.kenc(kpts0, data['scores0'])
        desc1 = desc1 + self.kenc(kpts1, data['scores1'])

        # Multi-layer Transformer network.
        desc0, desc1 = self.gnn(desc0, desc1)

        # Final MLP projection.
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)

        # Compute matching descriptor distance.
        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        scores = scores / self.config['descriptor_dim']**.5

        # Run the optimal transport.
        scores = log_optimal_transport(
            scores, self.bin_score, iters=self.config['sinkhorn_iterations'])

        # Get the matches with score above "match_threshold".
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = arange_like(indices0,
                              1)[None] == indices1.gather(1, indices0)
        mutual1 = arange_like(indices1,
                              1)[None] == indices0.gather(1, indices1)
        zero = scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
        valid0 = mutual0 & (mscores0 > self.config['match_threshold'])
        valid1 = mutual1 & valid0.gather(1, indices1)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

        return {
            'matches0': indices0,  # use -1 for invalid match
            'matches1': indices1,  # use -1 for invalid match
            'matching_scores0': mscores0,
            'matching_scores1': mscores1,
        }


class Matching(torch.nn.Module):
    """ Image Matching Frontend (SuperPoint + SuperGlue) """

    def __init__(self, config={}):
        super().__init__()
        self.superpoint = SuperPoint(config.get('superpoint', {}))
        self.superglue = SuperGlue(config.get('superglue', {}))

    def forward(self, data):
        """ Run SuperPoint (optionally) and SuperGlue
        SuperPoint is skipped if ['keypoints0', 'keypoints1'] exist in input
        Args:
          data: dictionary with minimal keys: ['image0', 'image1']
        """
        pred = {}

        # Extract SuperPoint (keypoints, scores, descriptors) if not provided
        if 'keypoints0' not in data:
            pred0 = self.superpoint({'image': data['image0']})
            pred = {**pred, **{k + '0': v for k, v in pred0.items()}}
        if 'keypoints1' not in data:
            pred1 = self.superpoint({'image': data['image1']})
            pred = {**pred, **{k + '1': v for k, v in pred1.items()}}

        # Batch all features
        # We should either have i) one image per batch, or
        # ii) the same number of local features for all images in the batch.
        data = {**data, **pred}

        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = torch.stack(data[k])

        # Perform the matching
        pred = {**pred, **self.superglue(data)}

        return pred


def points_to_homogeneous_representation(points: np.ndarray):
    homogenous_points = np.concatenate(
        (points, np.ones((points.shape[0], 1), dtype=points.dtype)), axis=1)
    return homogenous_points


def calculate_affine_transform(source_points: np.ndarray,
                               target_points: np.ndarray) -> np.ndarray:
    """
    TODO
    """
    transform, _, _, _ = np.linalg.lstsq(source_points,
                                         target_points,
                                         rcond=None)
    transform = transform.T
    return transform


def calculate_rigid_transform(source_points: np.ndarray,
                              target_points: np.ndarray) -> np.ndarray:
    """
    TODO
    """
    target = target_points.ravel()
    source_homogenous_points = points_to_homogeneous_representation(
        source_points)
    source = np.zeros((2 * source_points.shape[0], 2 * source_points.shape[1]))
    source[0::2, 0:target_points.shape[1] + 1] = source_homogenous_points[:, :]
    source[1::2, 0:target_points.shape[1] + 1] = source_homogenous_points[:, :]
    source[1::2,
           1], source[1::2,
                      0] = (-1) * source.copy()[1::2, 0], source.copy()[1::2,
                                                                        1]
    source[1::2, 2] = 0
    source[1::2, 3] = 1
    inv_source = np.linalg.pinv(source)
    params = inv_source @ target
    transform = np.array([
        [params[0], params[1], params[2]],
        [-params[1], params[0], params[3]],
        [0, 0, 1],
    ],
                         dtype=source_points.dtype)
    return transform


def perform_registration(source: torch.Tensor,
                         target: torch.Tensor,
                         superglue_params: Dict,
                         superpoint_weights_path: str,
                         superglue_weights_path: str,
                         transform_type: str,
                         device: str = "cuda") -> torch.Tensor:
    model = Matching(superglue_params).eval().to(device)
    model.superpoint.load_state_dict(
        torch.load(superpoint_weights_path, weights_only=True))
    model.superglue.load_state_dict(
        torch.load(superglue_weights_path, weights_only=True))

    source = source.to(device)
    target = target.to(device)

    pred = model({'image0': source, 'image1': target})
    pred = {k: v[0].detach().cpu().numpy() for k, v in pred.items()}
    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']

    #print(f"Number of source keypoints: {len(kpts0)}")
    #print(f"Number of target keypoints: {len(kpts1)}")
    matches = pred['matches0']

    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]

    if len(mkpts0) < 10:
        return np.eye(3), len(mkpts0), None
    #print(f"Number of matches: {len(h_pts0)}")
    transform = no_ransac(source_points=mkpts0,
                          target_points=mkpts1,
                          transform_type=transform_type)

    return transform, len(mkpts0), mkpts1


def ransac(source_points: np.ndarray,
           target_points: np.ndarray,
           num_iters: int = 500,
           threshold: float = 16.0,
           num_points: int = 3,
           transform_type: str = 'affine') -> np.ndarray:
    log.warning("Using ransac")
    best_transform = no_ransac(source_points,
                               target_points,
                               transform_type=transform_type)
    indices = np.arange(len(source_points))

    best_ratio = 0.0
    best_inliers = None

    for _ in range(num_iters):
        current_indices = np.random.choice(indices, num_points, replace=False)
        current_sp = source_points[current_indices, :]
        current_tp = target_points[current_indices, :]

        transform = no_ransac(current_sp,
                              current_tp,
                              transform_type=transform_type)

        transformed_target_points = (
            transform
            @ points_to_homogeneous_representation(target_points).swapaxes(
                1, 0)).swapaxes(0, 1)
        error = ((points_to_homogeneous_representation(source_points) -
                  transformed_target_points)**2).mean(axis=1)
        inliers = error < threshold
        ratio = inliers.sum() / len(source_points)
        if ratio > best_ratio:
            best_ratio = ratio
            best_transform = transform
            best_inliers = inliers
            #print(f"Current best ratio: {best_ratio}")

    if best_inliers is not None and np.any(best_inliers):
        best_transform = no_ransac(source_points[best_inliers, :],
                                   target_points[best_inliers, :],
                                   transform_type=transform_type)
    return best_transform


def no_ransac(source_points: np.ndarray,
              target_points: np.ndarray,
              transform_type: str = 'affine') -> np.ndarray:
    #print("noransac")
    if transform_type == "affine":
        h_pts0 = points_to_homogeneous_representation(source_points)
        h_pts1 = points_to_homogeneous_representation(target_points)
        transform = calculate_affine_transform(h_pts0, h_pts1)
    elif transform_type == "rigid":
        transform = calculate_rigid_transform(source_points, target_points)
    else:
        raise ValueError("Unsupported transform type (rigid or affine only).")
    return transform


def gaussian_smoothing(tensor: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Applies Gaussian smoothing to a given tensor.

    Args:
        tensor (torch.Tensor): The input tensor to smooth, with shape (C, H, W).
        sigma (float): The standard deviation of the Gaussian kernel.

    Returns:
        torch.Tensor: The smoothed tensor with the same shape as the input.
    """
    with torch.set_grad_enabled(False):
        kernel_size = int(sigma * 2.54) + 1 if int(
            sigma * 2.54) % 2 == 0 else int(sigma * 2.54)
        return torchvision.transforms.GaussianBlur(kernel_size,
                                                   sigma)(tensor.unsqueeze(0))


def get_combined_size(tensor_1: torch.Tensor,
                      tensor_2: torch.Tensor) -> Iterable[int]:
    """
    Retrieves the spatial sizes of two tensors.

    Args:
        tensor_1 (torch.Tensor): The first tensor, shape (..., H1, W1).
        tensor_2 (torch.Tensor): The second tensor, shape (..., H2, W2).

    Returns:
        Iterable[int]: A tuple of the sizes: (H1, W1, H2, W2).
    """
    tensor_1_y_size, tensor_1_x_size = tensor_1.size(-2), tensor_1.size(-1)
    tensor_2_y_size, tensor_2_x_size = tensor_2.size(-2), tensor_2.size(-1)
    return tensor_1_y_size, tensor_1_x_size, tensor_2_y_size, tensor_2_x_size


def calculate_resampling_ratio(x_sizes: Iterable, y_sizes: Iterable,
                               min_resolution: int) -> float:
    """Calculates the resampling ratio for resizing tensors.

    Args:
        x_sizes (Iterable): Iterable of width sizes for comparison.
        y_sizes (Iterable): Iterable of height sizes for comparison.
        min_resolution (int): Minimum resolution threshold.

    Returns:
        float: A common resampling ratio to scale tensors so that the larger
            image fits in a min_resxmin_res box.
    """
    x_size, y_size = max(x_sizes), max(y_sizes)
    min_size = min(x_size, y_size)
    if min_resolution > min_size:
        resampling_ratio = 1
    else:
        resampling_ratio = min_size / min_resolution
    return resampling_ratio


def resample(tensor: torch.Tensor,
             resample_ratio: float,
             mode: str = "bilinear") -> torch.Tensor:
    """Resamples a tensor to a lower resolution.

    Args:
        tensor (torch.Tensor): The input tensor with shape (C, H, W).
        resample_ratio (float): The scaling factor for resampling.
        mode (str): Interpolation mode, default is 'bilinear'.

    Returns:
        torch.Tensor: The resampled tensor.
    """
    return F.interpolate(tensor.unsqueeze(0),
                         scale_factor=1 / resample_ratio,
                         mode=mode,
                         recompute_scale_factor=False,
                         align_corners=False)


def initial_resampling(
    source: Union[torch.Tensor,
                  np.ndarray], target: Union[torch.Tensor,
                                             np.ndarray], resolution: int
) -> Tuple[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]]:
    """Resamples source and target tensors to a unified resolution.

    Args:
        source (Union[torch.Tensor, np.ndarray]): The source tensor or array.
        target (Union[torch.Tensor, np.ndarray]): The target tensor or array.
        resolution (int): The minimum resolution for resampling.

    Returns:
        Tuple[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]]:
        The resampled source, target, and the resampling ratio.
    """
    source_y_size, source_x_size, target_y_size, target_x_size = get_combined_size(
        source, target)
    resample_ratio = calculate_resampling_ratio((source_x_size, target_x_size),
                                                (source_y_size, target_y_size),
                                                resolution)
    resampled_source = resample(
        gaussian_smoothing(source, min(max(resample_ratio - 1, 0.1), 10)),
        resample_ratio)
    resampled_target = resample(
        gaussian_smoothing(target, min(max(resample_ratio - 1, 0.1), 10)),
        resample_ratio)
    return resampled_source, resampled_target, resample_ratio


def superpoint_superglue(src: torch.Tensor, trg: torch.Tensor,
                         registration_params: Dict) -> torch.Tensor:
    """
    Performs feature matching and transformation estimation using SuperPoint 
    and SuperGlue.

    Args:
        src (torch.Tensor): Source tensor with shape (C, H, W).
        trg (torch.Tensor): Target tensor with shape (C, H, W).
        registration_params (Dict): Parameters for registration.

    Returns:
        torch.Tensor: The transformed source, target, and registration info.
    """
    transform, num_matches, _ = perform_registration(src, trg,
                                                     **registration_params)

    return src, trg, transform, num_matches


def generate_rigid_matrix(angle: float, x0: float, y0: float, tx: float,
                          ty: float) -> np.ndarray:
    """
    Generates a rigid transformation matrix with rotation and translation.

    Args:
        angle (float): Rotation angle in degrees.
        x0 (float): X-coordinate of the center of rotation.
        y0 (float): Y-coordinate of the center of rotation.
        tx (float): Translation in the X direction.
        ty (float): Translation in the Y direction.

    Returns:
        np.ndarray: A 2x3 rigid transformation matrix.
    """
    angle = angle * np.pi / 180
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                [np.sin(angle),
                                 np.cos(angle), 0], [0, 0, 1]])
    cm1 = np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1]])
    cm2 = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]])
    translation_matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
    transform = cm1 @ rotation_matrix @ cm2 @ translation_matrix
    return transform[0:2, :]


def inverse_affine_transform(matrix: np.ndarray) -> np.ndarray:
    """
    Inverts a 2x3 affine transformation matrix.

    Args:
        matrix (np.ndarray): A 2x3 array representing the affine transform.
                             The first two columns represent the rotation and scaling,
                             and the last column represents the translation.

    Returns:
        np.ndarray: A 2x3 array representing the inverse affine transform.
    """
    if matrix.shape != (2, 3):
        raise ValueError("Input matrix must be of shape (2, 3)")

    # Extract the 2x2 rotation/scaling matrix and translation vector
    rotation_scaling = matrix[:, :2]
    translation = matrix[:, 2]

    # Compute the inverse of the 2x2 rotation/scaling matrix
    rotation_scaling_inv = np.linalg.inv(rotation_scaling)

    # Compute the inverse translation
    translation_inv = -np.dot(rotation_scaling_inv, translation)

    # Combine the inverse rotation/scaling and translation into a 2x3 matrix
    inverse_matrix = np.hstack(
        [rotation_scaling_inv,
         translation_inv.reshape(2, 1)])

    return inverse_matrix


def get_uniform_resizing_affine_matrix(scale: float) -> np.ndarray:
    """
    Generates a 2x3 affine transformation matrix for uniform resizing.
    
    Args:
        scale (float): Scaling factor for both axes (uniform scaling).
    
    Returns:
        np.ndarray: A 2x3 affine transformation matrix.
    """
    # Create the affine matrix
    affine_matrix = np.array(
        [
            [scale, 0, 0],  # Scale both x-axis
            [0, scale, 0]
        ],  # and y-axis
        dtype=np.float32)
    return affine_matrix


def adjust_matrix_and_size(image_shape, transform_matrix):
    """
    Adjust the transformation matrix and calculate the target size to ensure
    no part of the image is cut off during a rigid transformation.

    Parameters:
        image (np.ndarray): Input image.
        transform_matrix (np.ndarray): 2x3 affine transformation matrix.

    Returns:
        new_transform_matrix (np.ndarray): Adjusted 2x3 affine transformation matrix.
        new_width (int): Width of the output image.
        new_height (int): Height of the output image.
    """
    h, w = image_shape[0], image_shape[1]

    # Get the coordinates of the image corners
    corners = np.array([
        [0, 0, 1],  # Top-left
        [w, 0, 1],  # Top-right
        [0, h, 1],  # Bottom-left
        [w, h, 1]  # Bottom-right
    ])

    # Transform the corners
    transformed_corners = transform_matrix @ corners.T

    # Calculate the bounding box of the transformed image
    min_x = np.min(transformed_corners[0, :])
    max_x = np.max(transformed_corners[0, :])
    min_y = np.min(transformed_corners[1, :])
    max_y = np.max(transformed_corners[1, :])

    # New dimensions of the image
    new_width = int(np.ceil(max_x - min_x))
    new_height = int(np.ceil(max_y - min_y))

    # Adjust the translation part of the transformation matrix
    dx = -min_x
    dy = -min_y
    adjustment_matrix = np.array([[1, 0, dx], [0, 1, dy]])

    new_transform_matrix = adjustment_matrix @ np.vstack(
        [transform_matrix, [0, 0, 1]])

    return new_transform_matrix, (new_height, new_width)


def multi_feature(source: torch.Tensor,
                  target: torch.Tensor,
                  registration_params,
                  angle_step=30,
                  resolution=768) -> torch.Tensor:

    transforms = []
    source, target, resample_ratio = initial_resampling(
        source, target, resolution)

    resample_scale_matrix = get_uniform_resizing_affine_matrix(resample_ratio)

    for angle in range(-180, 180, angle_step):
        #print(angle)
        _, _, y_size, x_size = source.shape
        r_transform = generate_rigid_matrix(angle, x_size / 2, y_size / 2, 0,
                                            0)

        #transformed_source = cv2.warpAffine(
        #    np.array(source).squeeze().astype(np.float64),
        #    r_transform[:2, ...].squeeze(),
        #    np.array(source).squeeze().shape[::-1])

        r_transform, new_size_xy = adjust_matrix_and_size((y_size, x_size),
                                                          r_transform)
        transformed_source = cv2.warpAffine(
            np.array(source).squeeze().astype(np.float64),
            r_transform[:2, ...].squeeze(), new_size_xy)

        transformed_source = torch.tensor(
            transformed_source, dtype=torch.float).unsqueeze(0).unsqueeze(0)

        _, _, M, num_matches = superpoint_superglue(
            transformed_source,
            target,
            registration_params=registration_params)
        M = np.vstack(
            (resample_scale_matrix, np.array([[0, 0, 1]]))) @ M @ np.vstack(
                (r_transform, np.array([[0, 0, 1]]))) @ np.vstack(
                    (inverse_affine_transform(resample_scale_matrix),
                     np.array([[0, 0, 1]])))
        transforms.append((M, num_matches))

    best_matches = 0
    best_transform = np.eye(3)

    for transform, num_matches in transforms:
        if num_matches > best_matches:
            best_transform = transform
            best_matches = num_matches

    #(f"Final matches: {best_matches}")
    #print(f"Final transform: {best_transform}")
    return best_transform, source, target
