from typing import List, Optional, Callable, Dict
from collections import namedtuple
from itertools import chain
from functools import partial
import torch
from torch import nn, Tensor
import timm

from torchvision import models

from ts2.models.vit import get_vit_backbone
from ts2.models.resnet_backbone import get_resnet_backbone


def instantiate_backbone(which, params):
    if "resnet" in which:
        return get_resnet_backbone(which, params)
    elif "vit" in which:
        return get_vit_backbone(which, params)


class MLP(nn.Module):
    """MLP for classification head.
    Forward pass returns a tensor.
    """

    def __init__(self,
                 n_in: int,
                 hidden_layers: List[int],
                 n_out,
                 activation: str = "relu",
                 drop: float = 0.0,
                 bias: bool = True) -> None:
        super().__init__()
        layers_in = [n_in] + hidden_layers
        layers_out = hidden_layers + [n_out]

        act_callable = {
            "relu": nn.ReLU,
            "leaky_relu": nn.LeakyReLU,
            "prelu": nn.PReLU,
            "elu": nn.ELU,
            "gelu": nn.GELU
        }[activation]

        layers_list = []
        for i, (a, b) in enumerate(zip(layers_in, layers_out)):
            layers_list.append(nn.Linear(a, b, bias=bias))

            if i is not len(layers_in) - 1:
                layers_list.append(act_callable())

            if drop > 1e-6:
                layers_list.append(nn.Dropout(drop))

        self.layers = nn.Sequential(*layers_list)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self.layers.apply(init_weights)
        self.num_out = n_out

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class Classifier(nn.Module):
    """A network consists of a backbone and a classification head.
    Forward pass returns a dictionary, which contains both the logits and the
    embeddings.
    """

    def __init__(self, backbone_cf: Dict, head_params: Dict) -> None:
        """Initializes a Classifier.
        Args:
            backbone: the backbone, either a class, a function, or a parital.
                It defaults to resnet50.
            head: classification head to be attached to the backbone.
        """
        super().__init__()
        self.bb = instantiate_backbone(**backbone_cf)
        self.head = MLP(n_in=self.bb.num_out, **head_params)

    def forward(self, x: torch.Tensor) -> Dict:
        """Forward pass"""
        bb_out = self.bb(x)
        return {'logits': self.head(bb_out), 'embeddings': bb_out}


#def vit_backbone(params):
#    """Function used to call ViT model from PyTorch Image Models.
#    ViT source code in PyTorch Image Models (timm):
#    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py.
#    """
#    model = timm.create_model(**params)
#    model.head = nn.Identity()
#    model.num_out = model.embed_dim
#    return model

#class EvalNetwork(nn.Module):
#
#    def __init__(self, backbone_cf: Dict):
#        super().__init__()
#        self.bb = instantiate_backbone(**backbone_cf)
#
#    def forward(self, x, **kwargs):
#        return self.bb(x, **kwargs)


class ContrastiveLearningNetwork(torch.nn.Module):
    """A network consists of a backbone and projection head.
    Forward pass returns the normalized embeddings after a projection layer.
    """

    def __init__(self, backbone_cf: Dict, proj_params: Dict):
        super(ContrastiveLearningNetwork, self).__init__()
        self.bb = instantiate_backbone(**backbone_cf)
        self.proj = MLP(n_in=self.bb.num_out, **proj_params)
        self.num_out = self.proj.num_out

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        bb_out = self.bb(x, **kwargs)
        #bb_out_norm = torch.nn.functional.normalize(bb_out, p=2.0, dim=1)
        proj_out = self.proj(bb_out)
        proj_out_norm = torch.nn.functional.normalize(proj_out, p=2.0, dim=1)

        return {"emb": bb_out, "proj": proj_out_norm.unsqueeze(1)}


class VICRegNetwork(torch.nn.Module):
    """A network consists of a backbone and projection head."""

    def __init__(self, backbone_cf: Dict, proj_params: Dict):
        super(VICRegNetwork, self).__init__()
        self.bb = instantiate_backbone(**backbone_cf)
        self.proj = MLP(n_in=self.bb.num_out, **proj_params)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.proj(self.bb(x, **kwargs))


#class VICRegNetworkWithMask(torch.nn.Module):
#    """A network consists of a backbone and projection head."""
#
#    def __init__(self, backbone_cf: Dict, proj_params: Dict):
#        super(VICRegNetworkWithMask, self).__init__()
#        self.bb = instantiate_backbone(**backbone_cf)
#        self.proj = MLP(**proj_params)
#
#    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
#        return self.proj(self.bb(x, m))
#


class SimSiamNetwork(nn.Module):

    def __init__(self, backbone_cf: Dict, dim=384, pred_dim=96):
        super(SimSiamNetwork, self).__init__()
        self.bb = instantiate_backbone(**backbone_cf)

        # projection layer same as original paper
        prev_dim = self.bb.num_out
        self.proj = nn.Sequential(
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True),  # first layer
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True),  # second layer
            nn.Linear(prev_dim, dim, bias=True),
            nn.BatchNorm1d(dim, affine=False))  # output layer
        self.proj[
            6].bias.requires_grad = False  # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.pred = nn.Sequential(
            nn.Linear(dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),  # hidden layer
            nn.Linear(pred_dim, dim))  # output layer

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass, Takes in a single augmented image and runs a single branch of network, """
        bb_out = self.proj(self.bb(x, **kwargs))
        pred_out = self.pred(bb_out)

        return {'proj': bb_out, 'pred': pred_out}


class EMA():

    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(),
                                         ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


#class DeepDomainConfuser(Classifier):
#    """A model for deep domain confusion
#    A CNN but forward pass takes data from both source and target domain. Both
#    data are passed through the same backbone, bottleneck, and dense layers.
#    https://arxiv.org/abs/1412.3474
#    """
#
#    def __init__(self,
#                 backbone_cf: Dict[[], nn.Module],
#                 nc: int = 10,
#                 num_bottleneck: Optional[int] = None,
#                 pretrained: Optional[bool] = False) -> None:
#        """Initializes a DeepDomainConfusion network"""
#        raise NotImplementedError()
#        super().__init__(backbone, nc, num_bottleneck, pretrained)
#
#    def forward(self, src: Tensor, tgt: Tensor) -> Dict:
#        """Forward pass"""
#
#        src_bb_out = self.bb(src)
#        tgt_bb_out = self.bb(tgt)
#
#        return {
#            "src_logits": self.fc(self.bottleneck(src_bb_out["logits"])),
#            "src_embeddings": src_bb_out["embeddings"],
#            "tgt_logits": self.fc(self.bottleneck(tgt_bb_out["logits"])),
#            "tgt_embeddings": tgt_bb_out["embeddings"]
#        }
#

if __name__ == "__main__":
    #ddc = DeepDomainConfuser(backbone=models.resnet50, nc=5)
    from torchsrh.train.common import get_backbone
    from functools import partial
    # from torchsrh.models import resnet_backbone
    cf = {"backbone": {"which": "resnet50"}}
    simsiam = SimSiamNetwork(get_backbone({"backbone": {"which": "resnet50"}}))
    simclr = ContrastiveLearningNetwork(
        get_backbone(cf), partial(MLP, n_in=2048, hidden_layers=[], n_out=128))
    print(simclr)
