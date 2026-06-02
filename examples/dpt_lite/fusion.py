# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/antocad/FocusOnDepth and https://github.com/isl-org/DPT

# Fusion module for tactile transformer

import torch
import torch.nn as nn


class ResidualConvUnit(nn.Module):
    def __init__(self, features):
        super().__init__()

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: output
        """
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x


class Fusion(nn.Module):
    def __init__(self, resample_dim):
        super(Fusion, self).__init__()
        self.res_conv1 = ResidualConvUnit(resample_dim)
        self.res_conv2 = ResidualConvUnit(resample_dim)

    def forward(self, x, previous_stage=None):
        if previous_stage == None:
            previous_stage = torch.zeros_like(x)
        output_stage1 = self.res_conv1(x)
        output_stage1 += previous_stage
        output_stage2 = self.res_conv2(output_stage1)
        output_stage2 = nn.functional.interpolate(
            output_stage2, scale_factor=2, mode="bilinear", align_corners=True
        )
        return output_stage2
