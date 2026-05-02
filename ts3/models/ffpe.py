import numpy as np
import torch
import torch.nn as nn


class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, G: int, M: int, F_dim: int, H_dim: int, D: int, gamma: float):
        """
        THIS IS AUTHOR'S IMPL

        Learnable Fourier Features from https://arxiv.org/pdf/2106.02795.pdf (Algorithm 1)
        Implementation of Algorithm 1: Compute the Fourier feature positional encoding of a multi-dimensional position
        Computes the positional encoding of a tensor of shape [N, G, M]
        :param G: positional groups (positions in different groups are independent)
        :param M: each point has a M-dimensional positional values
        :param F_dim: depth of the Fourier feature dimension
        :param H_dim: hidden layer dimension
        :param D: positional encoding dimension
        :param gamma: parameter to initialize Wr
        """
        super().__init__()
        self.G = G
        self.M = M
        self.F_dim = F_dim
        self.H_dim = H_dim
        self.D = D
        self.gamma = gamma

        # Projection matrix on learned lines (used in eq. 2)
        self.Wr = nn.Linear(self.M, self.F_dim // 2, bias=False)
        # MLP (GeLU(F @ W1 + B1) @ W2 + B2 (eq. 6)
        self.mlp = nn.Sequential(
            nn.Linear(self.F_dim, self.H_dim, bias=True),
            nn.GELU(),
            nn.Linear(self.H_dim, self.D // self.G),
        )

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma**-2)

    def forward(self, x):
        """
        Produce positional encodings from x
        :param x: tensor of shape [N, G, M] that represents N positions where each position is in the shape of [G, M],
                  where G is the positional group and each group has M-dimensional positional values.
                  Positions in different positional groups are independent
        :return: positional encoding for X
        """
        N, G, M = x.shape
        # Step 1. Compute Fourier features (eq. 2)
        projected = self.Wr(x)
        cosines = torch.cos(projected)
        sines = torch.sin(projected)
        F = 1 / np.sqrt(self.F_dim) * torch.cat([cosines, sines], dim=-1)
        # Step 2. Compute projected Fourier features (eq. 6)
        Y = self.mlp(F)
        # Step 3. Reshape to x's shape
        PEx = Y.reshape((N, self.D))
        return PEx


class CenteredCoordinateFFPE(LearnableFourierPositionalEncoding):
    """
    SRH-specific wrapper that inherits from the author-style FFPE.

    Input:
        coords_xy: [N, 2], original coordinate units, e.g. pixels.

    Processing:
        1. Center by cell/tissue centroid.
        2. Divide by max_size / 2.
        3. Reshape to [N, G=1, M=2].
        4. Call LearnableFourierPositionalEncoding.forward().
    """

    def __init__(
        self,
        D: int,
        F_dim: int = 128,
        H_dim: int = 256,
        gamma: float = 2.0,
        max_size: float = 6000.0,
        center: bool = True,
    ):
        super().__init__(G=1, M=2, F_dim=F_dim, H_dim=H_dim, D=D, gamma=gamma)
        self.max_size = float(max_size)
        self.center = center
        self.coord_scale = max_size / 2.0 if center else max_size

    def preprocess_coords(self, coords_xy: torch.Tensor) -> torch.Tensor:
        if coords_xy.ndim != 2 or coords_xy.shape[-1] != 2:
            raise ValueError(
                f"Expected coords_xy [N, 2], got {tuple(coords_xy.shape)}."
            )

        coords = coords_xy.float()

        if self.center:
            coords = coords - coords.mean(dim=0, keepdim=True)
        coords = coords / self.coord_scale

        return coords.unsqueeze(1)  # Parent FFPE expects [N, G, M] = [N, 1, 2].

    def forward(self, coords_xy: torch.Tensor) -> torch.Tensor:
        return super().forward(self.preprocess_coords(coords_xy))
