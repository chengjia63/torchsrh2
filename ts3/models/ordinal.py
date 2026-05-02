import torch
from torch import nn


class MILOrdinalModel(nn.Module):
    def __init__(
        self,
        mil: nn.Module,
        head: nn.Module,
        pe: nn.Module | None = None,
    ):
        super().__init__()
        self.mil = mil
        self.head = head
        self.pe = pe

    def forward(self, embeddings: torch.Tensor, coords: torch.Tensor | None = None):
        if self.pe is not None:
            embeddings = embeddings + self.pe(
                coords.to(device=embeddings.device)
            ).to(dtype=embeddings.dtype)

        out = self.mil(embeddings)
        head_out = self.head(out["logits"])

        return {
            **out,
            **head_out,
        }


class SharedCORALHead(nn.Module):
    """
    Strict CORAL-style head:
    one shared scalar score + threshold-specific biases.

    Produces logits for:
        P(y > 0), P(y > 1), ..., P(y > K-2)
    """

    def __init__(self, num_classes: int):
        super().__init__()
        if num_classes < 2:
            raise ValueError(
                f"num_classes must be at least 2 for CORAL, got {num_classes}"
            )

        self.num_classes = num_classes
        self.coral_bias = nn.Parameter(torch.zeros(num_classes - 1))

    def forward(self, logits: torch.Tensor):
        score = logits.squeeze(-1)  # scalar or [B]
        logits = score.unsqueeze(-1) + self.coral_bias  # [K - 1] or [B, K - 1]

        return {
            "logits": logits,
            "score": score,
        }
