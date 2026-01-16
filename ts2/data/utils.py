import torch
import torchvision
import einops
from typing import List, Tuple
from ts2.data.mask_collator import MBMaskCollator
import time
import itertools
from dinov2.data.collate import OuterBiasedMasker,OuterCircularMasker


class SingleCellBlendedCollator(): # augmentation at training time - early exp

    def __init__(self, kernel_size, sigma, min_val_sigma, strong_transforms):
        self.ks = kernel_size
        self.sigma = sigma
        self.transform = strong_transforms
        self.min_val_sigma = min_val_sigma

    # def blend_batch(self, fg_im, bg_im):
    #     blend_scale = (fg_im[:, -1, ...] / 255).to(float)
    #     #blend_scale = torchvision.transforms.functional.gaussian_blur(
    #     #    blend_scale, kernel_size=self.ks, sigma= self.sigma)
    #     blend_scale = torch.cat([
    #         torchvision.transforms.functional.gaussian_blur(
    #             bs.unsqueeze(0), kernel_size=[self.ks, self.ks], sigma=[s, s])
    #         for bs, s in zip(
    #             blend_scale,
    #             torch.randn((len(blend_scale))).abs() * self.sigma + 1.0e-6)
    #     ])

    #     min_in = einops.reduce(
    #         blend_scale.masked_fill(~fg_im[:, -1, ...].to(bool), float("inf")),
    #         "b h w -> b", "min")
    #     min_in = min_in * torch.maximum(
    #         1 - torch.randn((len(blend_scale))).abs() * self.min_val_sigma,
    #         torch.tensor(1.0e-6))
    #     to_fill_idx = torch.where(
    #         blend_scale > einops.rearrange(min_in, "b -> b 1 1"))
    #     to_fill_value = min_in[to_fill_idx[0]]
    #     blend_scale[to_fill_idx[0], to_fill_idx[1],
    #                 to_fill_idx[2]] = to_fill_value

    #     blend_scale = blend_scale - einops.reduce(blend_scale,
    #                                               "b h w -> b 1 1", "min")
    #     blend_scale = blend_scale / einops.reduce(blend_scale,
    #                                               "b h w -> b 1 1", "max")
    #     blend_scale = torch.nan_to_num(blend_scale, nan=1.0)

    #     bg_im_cutout = torch.clone(bg_im[:, :-1, ...])

    #     bg_nu_mask = bg_im[:, -1, ...].to(bool).to(torch.uint8)
    #     bg_nu_mask3 = einops.repeat(bg_nu_mask,
    #                                 "b h w -> b c h w",
    #                                 c=bg_im_cutout.shape[1])
    #     bg_nu_fillval = (
    #         (bg_im_cutout * (1 - bg_nu_mask3)).sum(dim=(2, 3)) /
    #         einops.reduce(1 - bg_nu_mask, "b h w -> b 1", "sum")
    #     ).to(fg_im.dtype) # yapf: disable

    #     bg_nu_idx = torch.where(bg_nu_mask)
    #     bg_nu_fillval = bg_nu_fillval[bg_nu_idx[0]]
    #     bg_im_cutout[bg_nu_idx[0], :, bg_nu_idx[1],
    #                  bg_nu_idx[2]] = bg_nu_fillval

    #     reconstructed = (fg_im[:, :-1, ...] * blend_scale.unsqueeze(1) +
    #                      bg_im_cutout * (1 - blend_scale).unsqueeze(1)).to(
    #                          fg_im.dtype)

    #     return reconstructed, blend_scale

    def blend_batch(self, fg_im, bg_im):
        where_bm = [torch.stack(torch.where(b)) for b in fg_im[:, -1, ...] > 0]

        where_bm_min = [b.min(dim=1).values for b in where_bm]
        where_bm_max = [b.max(dim=1).values for b in where_bm]
        minrc = [(torch.randint(low=0,
                                high=torch.maximum(torch.tensor(1), i[0]),
                                size=(1, )),
                  torch.randint(low=0,
                                high=torch.maximum(torch.tensor(1), i[1]),
                                size=(1, ))) for i in where_bm_min]
        maxrc = [(torch.randint(low=torch.minimum(torch.tensor(63), i[0] + 1),
                                high=64,
                                size=(1, )),
                  torch.randint(low=torch.minimum(torch.tensor(63), i[1] + 1),
                                high=64,
                                size=(1, ))) for i in where_bm_max]

        #blend_scale = torch.zeros_like(fg_im)
        for i, (mi, ma, fg, bg) in enumerate(zip(minrc, maxrc, fg_im, bg_im)):
            bg[:, mi[0]:ma[0], mi[1]:ma[1]] = fg[:, mi[0]:ma[0], mi[1]:ma[1]]
        #    blend_scale[i,:, mi[0]:ma[0], mi[1]: ma[1]] = 1
        return
        #return blend_scale

    def __call__(self, raw_batch):
        all_images = torch.stack([i["image"]
                                  for i in raw_batch])  # b aug c h w
        assert all_images.shape[1] == 1
        fg = all_images[:, 0, ...]
        bg = torch.clone(all_images[:, 0, ...])
        bg = bg[torch.randperm(len(bg))]
        self.blend_batch(fg, bg)

        #fg_clone = torch.clone(fg[:, :3, ...])
        #bg_clone = torch.clone(bg[:, :3, ...])
        #t0 = time.time()
        #blend_scale = self.blend_batch(fg, bg)
        #t1 = time.time()
        #print(t1-t0)

        fg = torch.stack([self.transform(i) for i in fg[:, :3, ...]])
        bg = torch.stack([self.transform(i) for i in bg[:, :3, ...]])
        all_im = torch.stack((fg, bg), dim=1)

        #torch.save(
        #    {
        #       "blend_scale": blend_scale,
        #        "fg": fg,
        #        "alt_fg": bg,
        #        "fg_clean": fg_clone,
        #        "bg_clean": bg_clone
        #    }, "out.pt")
        #exit(1)
        return {
            "image": all_im,
            "label": torch.stack([i["label"] for i in raw_batch]),
            "path": [i["path"] for i in raw_batch]
        }

class SingleCellTokenBandShuffleCollator(): # used for pertubation evaluations

    def __init__(self, band_width, strong_transforms, use_mean=False):
        self.band_width = band_width
        self.transform = strong_transforms
        self.use_mean = use_mean

    def blend_batch(self, fg_im, bg_im):

        fg_mask = torch.zeros(fg_im.shape)
        fg_mask[:,:,self.band_width * 4:-self.band_width * 4, self.band_width * 4:-self.band_width * 4] = 1

        if self.use_mean:
            return (fg_im * (fg_mask) + 
                    bg_im.mean(dim=[2,3]).unsqueeze(2).unsqueeze(2) * (1-fg_mask))
        else:
            return fg_im * (fg_mask) + bg_im * (1-fg_mask)

    def __call__(self, raw_batch):
        all_images = torch.stack([i["image"]
                                  for i in raw_batch])  # b aug c h w
        assert all_images.shape[1] == 1
        fg = torch.clone(all_images[:, 0, ...])
        #fg_clean = torch.clone(all_images[:, 0, ...]) # for viz

        bg = torch.clone(all_images[:, 0, ...])
        bg = bg[torch.randperm(len(bg))]

        fg = self.blend_batch(fg, bg)

        fg = torch.stack([self.transform(i) for i in fg[:, :3, ...]])
        #bg = torch.stack([self.transform(i) for i in bg[:, :3, ...]])

        #torch.save({
        #        "alt_fg": fg,
        #        "bg": bg,
        #        "fg_clean": fg_clean,
        #    }, "band_perturb3.pt")
        #exit(0)

        return {
            "image": fg.unsqueeze(1),
            "label": torch.stack([i["label"] for i in raw_batch]),
            "path": [[i["path"][0] for i in raw_batch]]
        }

    #def __call__(self, raw_batch):
    #    import copy
    #    seed = 42
    #    torch.manual_seed(seed)
    #    perm = torch.randperm(len(raw_batch))
    #    for j in range(7):
    #        self.band_width = j+2
    #        all_images = torch.stack([i["image"]
    #                                  for i in copy.deepcopy(raw_batch)])  # b aug c h w
    #        assert all_images.shape[1] == 1
    #        fg = torch.clone(all_images[:, 0, ...])
    #        fg_clean = torch.clone(all_images[:, 0, ...]) # for viz
    #        bg = torch.clone(all_images[:, 0, ...])

    #        bg = bg[perm]

    #        fg = self.blend_batch(fg, bg)

    #        fg = torch.stack([self.transform(i) for i in fg[:, :3, ...]])
    #        bg = torch.stack([self.transform(i) for i in bg[:, :3, ...]])

    #        torch.save({
    #                "alt_fg": fg,
    #                "bg": bg,
    #                "fg_clean": fg_clean,
    #            }, f"band_perturb{j}.pt")
    #    exit(0)

    #    return {
    #        "image": fg.unsqueeze(1),
    #        "label": torch.stack([i["label"] for i in raw_batch]),
    #        "path": [[i["path"][0] for i in raw_batch]]
    #    }


class SingleCellTokenMaskShuffleCollator:  # used for perturbation evaluations
    def __init__(self,
                 mask_generator_which,
                 mask_generator_params,
                 strong_transforms,
                 use_mean: bool = False):
        """
        Args:
            mask_generator: callable with signature () -> 2D mask (H, W)
                Typically an instance of OuterCircularMasker.
            strong_transforms: augmentation pipeline applied to blended images.
            use_mean (bool): if True, background is replaced by per-image mean.
        """
        if mask_generator_which=="OuterBiasedMasker":
            self.mask_generator = OuterBiasedMasker(**mask_generator_params)
        elif mask_generator_which=="OuterCircularMasker":
            self.mask_generator = OuterCircularMasker(**mask_generator_params)
        else:
            assert False

        self.transform = strong_transforms
        self.use_mean = use_mean


    def blend_batch(self, fg_im, bg_im, fg_mask):
        """
        fg_im: (B, C, H, W)
        bg_im: (B, C, H, W)
        fg_mask: (B, H, W), uint8/bool, 1=fg, 0=bg
        """
        fg_mask = fg_mask.to(device=fg_im.device, dtype=fg_im.dtype)
        fg_mask = fg_mask.unsqueeze(1)                        # (B, 1, H, W)
        fg_mask = fg_mask.expand(-1, fg_im.shape[1], -1, -1)  # (B, C, H, W)

        if self.use_mean:
            bg_mean = bg_im.mean(dim=[2,3], keepdim=True)     # (B, C, 1, 1)
            return fg_im * fg_mask + bg_mean * (1 - fg_mask)
        else:
            return fg_im * fg_mask + bg_im * (1 - fg_mask)


    def _make_one_aug_view(self, fg: torch.Tensor) -> torch.Tensor:
        """
        Make a single augmented view:
          - shuffled background
          - mask generation
          - blend
          - strong transform

        fg: (B, C, H, W)
        returns: (B, C, H, W)
        """
        B, C, H, W = fg.shape
        device = fg.device

        # shuffled background
        bg = fg[torch.randperm(B, device=device)]

        # masks (B, H, W)
        masks = torch.stack(
            [1 - self.mask_generator().to(device) for _ in range(B)],
            dim=0,
        )

        blended = self.blend_batch(fg, bg, masks)  # (B, C, H, W)

        # apply strong transforms per sample
        return torch.stack(
            [self.transform(x) for x in blended[:, :3, ...]],
            dim=0,  # (B, C, H, W)
        )

    def __call__(self, raw_batch):
        # all_images: (B, 1, C, H, W)
        all_images = torch.stack([i["image"] for i in raw_batch])
        assert all_images.shape[1] == 1

        fg = all_images[:, 0, ...].clone()  # (B, C, H, W)
        view = self._make_one_aug_view(fg).unsqueeze(1)  # (B, 1, C, H, W)

        return {
            "image": view,
            "label": torch.stack([i["label"] for i in raw_batch]),
            "path": [[i["path"][0] for i in raw_batch]],
        }

class SingleCellTokenMaskShuffleMultiAugCollator(SingleCellTokenMaskShuffleCollator):
    def __init__(self, n_aug=2, **kwargs):
        super().__init__(**kwargs)
        self.n_aug = n_aug

    def __call__(self, raw_batch):
        # all_images: (B, 1, C, H, W)
        all_images = torch.stack([i["image"] for i in raw_batch])
        assert all_images.shape[1] == 1

        fg = all_images[:, 0, ...].clone()  # (B, C, H, W)

        views = [
            self._make_one_aug_view(fg).unsqueeze(1)
            for _ in range(self.n_aug)
        ]

        images = torch.cat(views, dim=1)  # (B, n_aug, C, H, W)

        return {
            "image": images,
            "label": torch.stack([i["label"] for i in raw_batch]),
            "path": [[i["path"][0] for i in raw_batch]],
        }

class CellMILCollator(object):
    def __init__(self):
        super(CellMILCollator, self).__init__()


    def __call__(self, batch):
        return {
            "path": [inst["path"] for inst in batch],
            "pixels": [inst["pixels"] for inst in batch],
            "label": torch.stack([i["label"] for i in batch])
        }


def get_collate_fn(which, params):
    collate_list = {
        "MBMaskCollator": MBMaskCollator,
        "SingleCellBlendedCollator": SingleCellBlendedCollator,
        "CellMILCollator": CellMILCollator,
        "SingleCellTokenBandShuffleCollator": SingleCellTokenBandShuffleCollator,
        "SingleCellTokenMaskShuffleCollator": SingleCellTokenMaskShuffleCollator,
        "SingleCellTokenMaskShuffleMultiAugCollator": SingleCellTokenMaskShuffleMultiAugCollator
    }
    return collate_list[which](**params)

defer_sxf_collators = {
    "SingleCellBlendedCollator",
    "SingleCellTokenBandShuffleCollator",
    "SingleCellTokenMaskShuffleCollator",
    "SingleCellTokenMaskShuffleMultiAugCollator"
}

