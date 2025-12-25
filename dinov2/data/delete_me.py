class OuterBiasedMasker():

    def __init__(self,
                 mask_size: int,
                 token_size: int,
                 dist_power: float = 2.0,
                 sharpness_range=[1, 5],
                 device='cpu',
                 dtype=torch.float32):
        """
        Args:
            mask_size (int): size of the square mask (mask_size x mask_size).
            power (float): strength of the bias toward outer elements.
            device (str): device to use for tensors ('cpu' or 'cuda').
            dtype (torch.dtype): dtype for probability tensor.
        """
        self.mask_size = mask_size
        self.token_size = token_size
        self.device = device
        self.dtype = dtype
        self.sharpness_range = sharpness_range

        # Precompute distance
        x = torch.arange(mask_size, device=device, dtype=dtype)
        y = torch.arange(mask_size, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(x, y, indexing='ij')

        cx, cy = (mask_size - 1) / 2, (mask_size - 1) / 2
        dist2 = (xx - cx)**dist_power + (yy - cy)**dist_power
        self.norm_dist = dist2 / dist2.max()  # normalize to [0, 1]
        self.upsample_kernel = torch.ones((token_size, token_size),
                                          dtype=torch.uint8,
                                          device=device)

    def __call__(self, num_masked: int) -> torch.Tensor:
        """
        Sample a k x k binary mask with num_masked entries = 1, biased toward outer region.

        Args:
            num_masked (int): number of tokens to mask.

        Returns:
            torch.Tensor: mask of shape (k, k), dtype=torch.uint8.
        """
        sharpness = random.uniform(*self.sharpness_range)
        prob = self.norm_dist.pow(sharpness)
        prob = (prob / prob.sum()).flatten()
        flat_mask = torch.zeros(self.mask_size * self.mask_size,
                                dtype=torch.uint8,
                                device=self.device)
        idx = torch.multinomial(prob,
                                num_samples=num_masked,
                                replacement=False)
        flat_mask[idx] = 1
        base_mask = flat_mask.view(self.mask_size, self.mask_size)
        return torch.kron(base_mask, self.upsample_kernel)


class GaussianNoise(torch.nn.Module):
    """object to add guassian noise to images."""

    def __init__(self, min_var: float = 0.01, max_var: float = 0.1):
        super().__init__()
        self.min_var = min_var
        self.max_var = max_var

    def forward(self, tensor):
        min_, max_ = tensor.min(), tensor.max()
        var = random.uniform(self.min_var, self.max_var)
        noisy = tensor + torch.randn(tensor.size()) * var
        return torch.clamp(noisy, min=min_, max=max_)


class CellCollator():

    def __init__(self,
                 mask_ratio_tuple,
                 mask_probability,
                 dtype,
                 omb_params,
                 num_token_swap,
                 n_tokens=None,
                 mask_generator=None):
        self.mask_ratio_tuple = mask_ratio_tuple
        self.mask_probability = mask_probability
        self.dtype = dtype
        self.n_tokens = n_tokens
        self.mask_generator = mask_generator

        self.color_jittering = transforms.Compose([
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
                ],
                p=0.8,
            ),
            transforms.RandomApply([GaussianNoise(min_var=0, max_var=0.05)],
                                   p=0.5),
            GaussianBlur(p=0.5, radius_max=1),
            transforms.RandomSolarize(threshold=0.5, p=.2),
            transforms.RandomGrayscale(p=0.2),
        ])
        self.obm = OuterBiasedMasker(**omb_params)
        self.num_token_swap = num_token_swap

    def __call__(self, samples_list):
        n_global_crops = len(samples_list[0][0]["global_crops"])
        n_local_crops = len(samples_list[0][0]["local_crops"])

        collated_global_crops = torch.stack([
            s[0]["global_crops"][i] for i in range(n_global_crops)
            for s in samples_list
        ])
        collated_local_crops = torch.stack([
            s[0]["local_crops"][i] for i in range(n_local_crops)
            for s in samples_list
        ])

        bg_swap_masks = torch.stack([
            self.obm(self.num_token_swap)
            for _ in range(len(collated_local_crops))
        ]).unsqueeze(1)
        collated_local_crops = (
            collated_local_crops *
            (1 - bg_swap_masks) + collated_local_crops[torch.randperm(
                collated_local_crops.size(0))] * bg_swap_masks)
        collated_local_crops = torch.stack(
            [self.color_jittering(i) for i in collated_local_crops])

        images = {
            "collated_global_crops": collated_global_crops.to(self.dtype),
            "collated_local_crops": collated_local_crops.to(self.dtype),
            "bg_swap_masks": bg_swap_masks,
        }
        B = len(collated_global_crops)
        N = self.n_tokens
        images.update(
            create_mask(B, N, self.mask_ratio_tuple, self.mask_probability,
                        self.mask_generator))

        return images
