"""Thin ts3 adapter around the vendored PANTHER set encoder.

The PANTHER MAP-EM aggregator, DirNIW prior, and ``allcat`` output are imported
verbatim from the ``panther`` package. This module only adds:

  - per-bag forward conforming to ``SlideABMILOrdinalModule``'s contract
    (single ``[N, D]`` input, returns ``{logits, score, attention}``)
  - a CORAL-compatible regressor + head wrapper.
"""

from typing import Optional

import torch
from torch import nn

from panther.layers import PANTHERBase


class PantherOrdinalModel(nn.Module):
    """PANTHER set encoder + scalar regressor + CORAL head.

    The existing ``SlideABMILOrdinalModule`` meta_arch loops over bags and
    passes one slide's ``embeddings`` ``[N, D]`` per call. We unsqueeze to
    ``[1, N, D]`` for ``PANTHERBase``, take the ``allcat`` flat representation,
    project it to a scalar with ``regressor``, then apply the CORAL ``head``.

    The placeholder ``attention`` returned here is the per-cell sum of the
    final EM responsibilities (``qq.sum(-1)``) — a useful scalar per cell that
    keeps the meta_arch's predict-step plumbing intact.
    """

    def __init__(
        self,
        encoder: PANTHERBase,
        regressor: nn.Module,
        head: nn.Module,
    ):
        super().__init__()
        self.encoder = encoder
        self.regressor = regressor
        self.head = head

    def forward(
        self, embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> dict:

        if embeddings.ndim != 2:
            raise ValueError(
                f"PantherOrdinalModel expects [N, D] embeddings, got "
                f"{tuple(embeddings.shape)}"
            )
        S = embeddings.unsqueeze(0)  # [1, N, D]
        out, qqs = self.encoder(S, mask=mask)  # out: [1, F]
        score_logit = self.regressor(out.squeeze(0))  # [1]
        head_out = self.head(score_logit)
        # qqs from PANTHERBase is [B, N, K, H] (H=1).
        attention = qqs[..., 0].sum(dim=-1)  # [1, N]
        return {
            "logits": head_out["logits"],
            "score": head_out["score"],
            "attention": attention,
        }
