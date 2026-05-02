import torch
from torch import nn
import torch.nn.functional as F


class CORALLoss(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.long().view(-1)
        thresholds = torch.arange(self.num_classes - 1, device=labels.device)
        levels = (labels.unsqueeze(1) > thresholds.unsqueeze(0)).to(dtype=logits.dtype)
        return F.binary_cross_entropy_with_logits(logits, levels)
