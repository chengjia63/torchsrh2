from collections.abc import Callable

import torch
from torch import nn
import torch.nn.functional as F


class MLPRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (),
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()

        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim, bias=bias),
                    act_layer(),
                    nn.Dropout(drop),
                ]
            )
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1, bias=bias))

        self.layers = nn.Sequential(*layers)

    def forward(self, embeddings: torch.Tensor):
        return self.layers(embeddings)


class GatedABMIL(nn.Module):
    def __init__(
        self,
        instance_encoder: nn.Module,
        attention_v: nn.Module,
        attention_u: nn.Module,
        attention_w: nn.Module,
        regressor: nn.Module,
    ):
        super().__init__()
        self.instance_encoder = instance_encoder
        self.attention_v = attention_v
        self.attention_u = attention_u
        self.attention_w = attention_w
        self.regressor = regressor

    def forward(self, embeddings: torch.Tensor):
        encoded = self.instance_encoder(embeddings)  # [N, H]

        attention_logits = self.attention_w(
            self.attention_v(encoded) * self.attention_u(encoded)
        ).transpose(
            1, 0
        )  # [1, N]

        attention = F.softmax(attention_logits, dim=1)  # [1, N]
        pooled = torch.mm(attention, encoded).squeeze(0)  # [H]
        logits = self.regressor(pooled)  # [1]

        return {
            "attention": attention.squeeze(0),
            "encoded_embeddings": encoded,
            "logits": logits,
        }


class GatedABMILRegThenAgg(nn.Module):
    def __init__(
        self,
        instance_encoder: nn.Module,
        attention_v: nn.Module,
        attention_u: nn.Module,
        attention_w: nn.Module,
        regressor: nn.Module,
    ):
        super().__init__()
        self.instance_encoder = instance_encoder
        self.attention_v = attention_v
        self.attention_u = attention_u
        self.attention_w = attention_w
        self.regressor = regressor

    def forward(self, embeddings: torch.Tensor):
        encoded = self.instance_encoder(embeddings)  # [N, H]

        attention_logits = self.attention_w(
            self.attention_v(encoded) * self.attention_u(encoded)
        ).transpose(
            1, 0
        )  # [1, N]

        attention = F.softmax(attention_logits, dim=1)  # [1, N]
        regressed = self.regressor(encoded)  # [N, 1]
        logits = torch.mm(attention, regressed).squeeze(0)  # [1]

        return {
            "attention": attention.squeeze(0),
            "regressed_embeddings": regressed,
            "logits": logits,
        }
