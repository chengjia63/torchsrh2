import torch


class CoordBaseTransforms:
    """
    Always-applied coordinate preprocessing.

    Reads:
        sample["coords"]: [N, 2]

    Writes:
        sample["coords"]: [N, 1, 2]
    """

    def __init__(
        self,
        max_size: float = 6000.0,
        center: bool = True,
    ):
        self.max_size = float(max_size)
        self.center = center
        self.coord_scale = self.max_size / 2.0 if center else self.max_size

    def __call__(self, sample: dict) -> dict:
        out = dict(sample)

        coords = out["coords"]
        if not torch.is_tensor(coords):
            coords = torch.as_tensor(coords)

        coords = coords.float()

        if coords.ndim != 2 or coords.shape[-1] != 2:
            raise ValueError(
                f'Expected sample["coords"] with shape [N, 2], got {tuple(coords.shape)}.'
            )

        coords_ffpe = coords
        if self.center:
            coords_ffpe = coords_ffpe - coords_ffpe.mean(dim=0, keepdim=True)

        coords_ffpe = coords_ffpe / self.coord_scale
        out["coords"] = coords_ffpe.unsqueeze(1)

        return out


class CellDropAugment:
    """
    Train-only cell dropping augmentation.

    Randomly discards a fraction of cells, updating both embeddings and coords
    consistently.

    Reads/writes:
        sample["embeddings"]: [N, D]
        sample["coords"]: [N, 2]

    Intended to run before CoordBaseTransforms.
    """

    def __init__(
        self,
        drop_rate: list[float] = [0.0, 0.5],
        min_cells: int = 1,
        p: float = 0.5,
    ):
        if len(drop_rate) != 2 or not (0.0 <= drop_rate[0] <= drop_rate[1] < 1.0):
            raise ValueError(
                f"drop_rate must be [min, max] with 0 <= min <= max < 1, got {drop_rate}."
            )
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}.")
        self.drop_rate_min = float(drop_rate[0])
        self.drop_rate_range = float(drop_rate[1]) - float(drop_rate[0])
        self.min_cells = int(min_cells)
        self.p = float(p)

    def __call__(self, sample: dict) -> dict:
        if torch.rand(()) >= self.p:
            return sample

        out = dict(sample)
        embeddings = out["embeddings"]
        coords = out["coords"]

        if not torch.is_tensor(embeddings):
            embeddings = torch.as_tensor(embeddings)
        if not torch.is_tensor(coords):
            coords = torch.as_tensor(coords)

        rate = self.drop_rate_min + torch.rand(()).item() * self.drop_rate_range
        n = embeddings.shape[0]
        keep_n = max(self.min_cells, int(round(n * (1.0 - rate))))

        if keep_n >= n:
            return out

        keep_idx = torch.randperm(n, device=embeddings.device)[:keep_n].sort().values
        out["embeddings"] = embeddings[keep_idx]
        out["coords"] = coords[keep_idx]
        return out


class StrongCoordAugment:
    """
    Train-only coordinate augmentation.

    Reads/writes:
        sample["coords"]: [N, 2]

    Intended to run before PrepareFFPECoords.
    """

    def __init__(
        self,
        flip: bool = True,
        transpose: bool = True,
        jitter_std: float = 12,
        random_offset: bool = True,
        max_size: float = 6000.0,
        p: float = 0.5,
    ):
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}.")

        self.flip = flip
        self.transpose = transpose
        self.jitter_std = float(jitter_std)
        self.random_offset = random_offset
        self.max_size = float(max_size)
        self.p = float(p)

    def __call__(self, sample: dict) -> dict:
        out = dict(sample)

        coords = out["coords"]
        if not torch.is_tensor(coords):
            coords = torch.as_tensor(coords)
        coords = coords.float()

        if coords.ndim != 2 or coords.shape[-1] != 2:
            raise ValueError(
                f'Expected sample["coords"] with shape [N, 2], got {tuple(coords.shape)}.'
            )

        if coords.shape[0] == 0:
            out["coords"] = coords
            return out

        center = coords.mean(dim=0, keepdim=True)
        rel = coords - center

        if self.flip:
            signs = torch.where(
                torch.rand(2, device=coords.device) < self.p,
                -1.0,
                1.0,
            ).to(dtype=coords.dtype)
            rel = rel * signs

        if self.transpose and torch.rand((), device=coords.device) < self.p:
            rel = rel.flip(-1)  # [x, y] -> [y, x]

        coords_aug = rel + center

        if self.random_offset and torch.rand((), device=coords.device) < self.p:
            xy_min = coords_aug.amin(dim=0, keepdim=True)
            wh = coords_aug.amax(dim=0, keepdim=True) - xy_min

            new_min = torch.rand_like(xy_min) * (self.max_size - wh).clamp_min(0.0)
            coords_aug = coords_aug + new_min - xy_min

        if self.jitter_std > 0 and torch.rand((), device=coords.device) < self.p:
            coords_aug = coords_aug + torch.randn_like(coords_aug) * self.jitter_std

        out["coords"] = coords_aug
        return out
