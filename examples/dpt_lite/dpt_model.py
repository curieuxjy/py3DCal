# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/antocad/FocusOnDepth and https://github.com/isl-org/DPT

# Tactile transformer model

import numpy as np
import timm
import torch.nn as nn

from .fusion import Fusion
from .head import HeadDepth, HeadSeg
from .reassemble import Reassemble


class DPTModel(nn.Module):
    def __init__(
        self,
        image_size=(3, 384, 384),
        patch_size=16,
        emb_dim=1024,
        resample_dim=256,
        read="projection",
        num_layers_encoder=24,
        hooks=[5, 11, 17, 23],
        reassemble_s=[4, 8, 16, 32],
        transformer_dropout=0,
        nclasses=2,
        type="full",
        model_timm="vit_large_patch16_384",
        pretrained=False,
    ):
        """
        type : {"full", "depth", "segmentation"}
        image_size : (c, h, w)
        patch_size : *a square*
        emb_dim <=> D (in the paper)
        resample_dim <=> ^D (in the paper)
        read : {"ignore", "add", "projection"}
        """
        super().__init__()

        self.transformer_encoders = timm.create_model(model_timm, pretrained=pretrained)
        self.type_ = type

        # Register hooks
        self.activation = {}
        self.hooks = hooks
        self._get_layers_from_hooks(self.hooks)

        # Reassembles Fusion
        self.reassembles = []
        self.fusions = []
        for s in reassemble_s:
            self.reassembles.append(
                Reassemble(image_size, read, patch_size, s, emb_dim, resample_dim)
            )
            self.fusions.append(Fusion(resample_dim))
        self.reassembles = nn.ModuleList(self.reassembles)
        self.fusions = nn.ModuleList(self.fusions)

        # Head
        if type == "full":
            self.head_depth = HeadDepth(resample_dim)
            self.head_segmentation = HeadSeg(resample_dim, nclasses=nclasses)
        elif type == "depth":
            self.head_depth = HeadDepth(resample_dim)
            self.head_segmentation = None
        else:
            self.head_depth = None
            self.head_segmentation = HeadSeg(resample_dim, nclasses=nclasses)

    def forward(self, img):

        t = self.transformer_encoders(img)
        previous_stage = None
        for i in np.arange(len(self.fusions) - 1, -1, -1, dtype=int):
            hook_to_take = "t" + str(self.hooks[int(i)])
            activation_result = self.activation[hook_to_take]
            reassemble_result = self.reassembles[i](activation_result)
            fusion_result = self.fusions[i](reassemble_result, previous_stage)
            previous_stage = fusion_result
        out_depth = None
        out_segmentation = None
        if self.head_depth != None:
            out_depth = self.head_depth(previous_stage)
        if self.head_segmentation != None:
            out_segmentation = self.head_segmentation(previous_stage)
        return out_depth, out_segmentation

    def _get_layers_from_hooks(self, hooks: list):
        def get_activation(name):
            def hook(model, input, output):
                self.activation[name] = output

            return hook

        for h in hooks:
            self.transformer_encoders.blocks[h].register_forward_hook(
                get_activation("t" + str(h))
            )
