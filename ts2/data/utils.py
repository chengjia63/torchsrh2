import torch
import torchvision
import einops
from typing import List, Tuple
from ts2.data.mask_collator import MBMaskCollator
import time


class SingleCellBlendedCollator():

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
        "CellMILCollator": CellMILCollator
    }
    return collate_list[which](**params)

