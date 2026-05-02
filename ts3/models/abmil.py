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
        input_dim: int,
        hidden_dim: int,
        final_dim: int,
        dropout: float = 0.0,
        regressor_hidden_dims: tuple[int, ...] = (),
        regressor_act_layer: Callable[..., nn.Module] = nn.GELU,
        regressor_drop: float = 0.0,
        regressor_bias: bool = True,
    ):
        super().__init__()
        self.instance_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attention_v = nn.Sequential(
            nn.Linear(hidden_dim, final_dim),
            nn.Tanh(),
        )
        self.attention_u = nn.Sequential(
            nn.Linear(hidden_dim, final_dim),
            nn.Sigmoid(),
        )
        self.attention_w = nn.Linear(final_dim, 1)
        self.regressor = MLPRegressor(
            hidden_dim,
            hidden_dims=regressor_hidden_dims,
            act_layer=regressor_act_layer,
            drop=regressor_drop,
            bias=regressor_bias,
        )

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
        input_dim: int,
        hidden_dim: int,
        final_dim: int,
        dropout: float = 0.0,
        regressor_hidden_dims: tuple[int, ...] = (),
        regressor_act_layer: Callable[..., nn.Module] = nn.GELU,
        regressor_drop: float = 0.0,
        regressor_bias: bool = True,
    ):
        super().__init__()
        self.instance_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attention_v = nn.Sequential(
            nn.Linear(hidden_dim, final_dim),
            nn.Tanh(),
        )
        self.attention_u = nn.Sequential(
            nn.Linear(hidden_dim, final_dim),
            nn.Sigmoid(),
        )
        self.attention_w = nn.Linear(final_dim, 1)
        self.regressor = MLPRegressor(
            hidden_dim,
            hidden_dims=regressor_hidden_dims,
            act_layer=regressor_act_layer,
            drop=regressor_drop,
            bias=regressor_bias,
        )

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
