from typing import Optional
import math

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer

from timm.layers import trunc_normal_

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads import UPerHead
from mmseg.models.losses import accuracy
from mmseg.utils import SampleList


@HEADS.register_module()
class UPerUpscale(UPerHead):
    def __init__(
        self,
        upscale_method: str = "transpose_conv",
        upscale_factor: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self_upscale_method = upscale_method
        self.upscale_factor = upscale_factor
        self.upscale, self.conv_seg = self._build_upscale_and_head(upscale_factor, upscale_method)

    def _build_upscale_and_head(self, upscale_factor, upscale_method):

        upscale_layers = []

        in_channels = self.channels

        for i in range(upscale_factor):

            out_channels = in_channels // 2

            if upscale_method == "transpose_conv":
                upscale_layers.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=1),
                        nn.SyncBatchNorm(out_channels),
                        nn.ReLU(inplace=True),
                    )
                )

            elif upscale_method == "transpose_conv_conv":
                upscale_layers.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2, padding=0),
                        nn.SyncBatchNorm(in_channels),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                        nn.SyncBatchNorm(out_channels),
                    )
                )
            elif upscale_method == "resize_conv":
                upscale_layers.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=2, mode="nearest"),
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                        nn.SyncBatchNorm(out_channels),
                        nn.ReLU(inplace=True),
                    )
                )
            elif upscale_method == "jbu":
                raise NotImplementedError("Joint Bilateral Upsampling is not implemented yet.")
            else:
                raise ValueError(f"Upscale method not implemented: {upscale_method}")

            in_channels = out_channels

        upscale_layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))

        # Modify segmentation/prediction head, self.out_channels is the number of classes
        head = nn.Conv2d(out_channels, self.out_channels, kernel_size=1)

        return nn.Sequential(*upscale_layers), head

    def forward(self, inputs):
        output = self._forward_feature(inputs)
        output = self.upscale(output)
        output = self.conv_seg(output)

        return output
