"""Resnet model.

Adapted from torchvision. See THIRD_PARTY for third party license info.
https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

Copyright (c) 2022 Machine Learning in Neurosurgery Laboratory.
All rights reserved.
"""

from typing import Type, Any, Callable, Union, List, Optional, Dict

import torch
import torchvision
from torch import nn, Tensor


def conv3x3(in_planes: int,
            out_planes: int,
            stride: int = 1,
            groups: int = 1,
            dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                "BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3
    # convolution(self.conv2) while original implementation places the stride
    # at the first 1x1 convolution(self.conv1) according to "Deep residual
    # learning for image recognition"https://arxiv.org/abs/1512.03385. This
    # variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetBackbone(nn.Module):
    """A ResNet backbone model.

    ResNet architecture based on torchvision implementation. It does not
    include the dense fc layer at the end. The forward function returns the
    final latent representations.
    """

    def __init__(self,
                 num_channel_in: int,
                 in_planes: int,
                 layer_planes: List[int],
                 block: Type[Union[BasicBlock, Bottleneck]],
                 layers: List[int],
                 stride: Optional[Dict[str, int]] = None,
                 zero_init_residual: bool = False,
                 groups: int = 1,
                 width_per_group: int = 64,
                 replace_stride_with_dilation: Optional[List[bool]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 conv1_kernel_size: int = 7,
                 conv1_padding: int = 3,
                 do_maxpool: bool = True) -> None:

        super(ResNetBackbone, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = in_planes
        self.dilation = 1

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(
                                 replace_stride_with_dilation))

        if not stride:
            stride = {
                "conv1": 2,
                "maxpool": 2,
                "conv3_1": 2,
                "conv4_1": 2,
                "conv5_1": 2
            }
        self.groups = groups
        self.base_width = width_per_group

        # ----------------------------------------------------------------------
        # make layers
        self.conv1 = nn.Conv2d(num_channel_in,
                               self.inplanes,
                               kernel_size=conv1_kernel_size,
                               stride=stride["conv1"],
                               padding=conv1_padding,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        if do_maxpool:
            self.maxpool = nn.MaxPool2d(kernel_size=3,
                                        stride=stride["maxpool"],
                                        padding=1)
        else:
            self.maxpool = nn.Identity()

        self.layer1 = self._make_layer(block, layer_planes[0], layers[0])

        self.layer2 = self._make_layer(block,
                                       layer_planes[1],
                                       layers[1],
                                       stride=stride["conv3_1"],
                                       dilate=replace_stride_with_dilation[0])

        self.layer3 = self._make_layer(block,
                                       layer_planes[2],
                                       layers[2],
                                       stride=stride["conv4_1"],
                                       dilate=replace_stride_with_dilation[1])

        self.layer4 = self._make_layer(block,
                                       layer_planes[3],
                                       layers[3],
                                       stride=stride["conv5_1"],
                                       dilate=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.num_out = block.expansion * layer_planes[-1]
        # ----------------------------------------------------------------------
        # init layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each
        # residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type:ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type:ignore[arg-type]

    def _make_layer(self,
                    block: Type[Union[BasicBlock, Bottleneck]],
                    planes: int,
                    blocks: int,
                    stride: int = 1,
                    dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation,
                      norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Dict:
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x1 = self.maxpool(x0)

        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        x6 = self.avgpool(x5)
        x6 = torch.flatten(x6, 1)

        return x6

    def forward(self, x: Tensor) -> Dict:
        return self._forward_impl(x)


class ResNetSpatialBackbone(ResNetBackbone):

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def _forward_impl(self, x: Tensor, mask: Tensor) -> Dict:
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x1 = self.maxpool(x0)

        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        with torch.no_grad():
            mask = torchvision.transforms.functional.resize(
                mask, x5.shape[-2:])
            #mask = mask + 0.1 * (1 - mask)
        x5_ = x5 * mask

        x6 = self.avgpool(x5_)
        x6 = torch.flatten(x6, 1)

        return x6

    def forward(self, x: Tensor, mask: Tensor) -> Dict:
        return self._forward_impl(x, mask)


def get_resnet_backbone(which: str = "resnet50",
                        params: Dict = {}) -> ResNetBackbone:
    """Creates a resnet backbone."""
    blocks = {
        'resnet18': BasicBlock,
        'resnet34': BasicBlock,
        'resnet50': Bottleneck,
        'resnet101': Bottleneck,
        'resnet152': Bottleneck,
    }

    layers = {
        'resnet18': [2, 2, 2, 2],
        'resnet34': [3, 4, 6, 3],
        'resnet50': [3, 4, 6, 3],
        'resnet101': [3, 4, 23, 3],
        'resnet152': [3, 8, 36, 3],
    }

    default_params = {
        "block": blocks[which],
        "layers": layers[which],
        "num_channel_in": 3,
        "in_planes": 64,
        "layer_planes": [64, 128, 256, 512],
    }

    default_params.update(params)
    cell_mask = default_params.pop("cell_mask", False)

    if cell_mask:
        return ResNetSpatialBackbone(**default_params)
    else:
        return ResNetBackbone(**default_params)
