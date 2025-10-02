#
# COPYRIGHT (c) 2024 - Denso ADAS Engineering Services GmbH, Apache License 2.0
# Author: Zeeshan Khan Suri (z.suri@eu.denso.com)
#
# Wrapper for DINOv2 Vision Transformer backbone which is compatible with mmsegmentation >= 1.0

import copy
import os

from dinov2.models.vision_transformer import DinoVisionTransformer, Block, MemEffAttention

from typing import Optional, List, Tuple
from functools import partial
import warnings

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmseg.registry import MODELS


@MODELS.register_module()
class DinoVisionBackbone(DinoVisionTransformer, BaseModule):
    """mmsegmentation compatible Vision Transformer backbone.

    Inputs:
        size (str): size of ViT backbone. 'small', 'base', 'large', 'giant'
        freeze_vit (bool): Freezes the entire backbone.
            Default: False
        pretrained (str, optional): model pretrained path. (deprecated, use init_cfg instead)
            Default: None.
        init_cfg (dict, optional): Initialization config dict.
            Default: None.
        args, kwargs: Additional args that are passed to DinoVisionTransformer
    """

    # out_indices come from the DINOv2 configs, \
    # for e.g. https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_voc2012_ms_config.py
    default_out_indices = dict(
        small=[8, 9, 10, 11], base=[8, 9, 10, 11], large=[20, 21, 22, 23], giant=[36, 37, 38, 39]
    )

    def __init__(
        self,
        size: str = "base",
        freeze_vit: bool = False,
        init_cfg: Optional[dict] = None,
        out_indices: Optional[List[int]] = None,
        in_channels: Optional[int] = 3,
        patch_size: int = 16,
        fpn_scaling: Optional[int] = False,
        reuse_fpn_weights_from_reconstruction_head: bool = False,
        unwrap_checkpoint: bool = True,
        rescale_representations: bool = False,
        init_values: float = 1.0,
        *args,
        **kwargs,
    ):
        # Update DinoVisionTransformer arguments based on model size
        if "small" in size:
            vit_kwargs = dict(embed_dim=384, depth=12, num_heads=6)
        elif "base" in size:
            vit_kwargs = dict(embed_dim=768, depth=12, num_heads=12)
        elif "large" in size:
            vit_kwargs = dict(embed_dim=1024, depth=24, num_heads=16)
        elif "giant" in size:
            vit_kwargs = dict(embed_dim=1536, depth=40, num_heads=24)
        else:
            raise NotImplementedError("Choose size from 'small', 'base', 'large', 'giant'")

        kwargs.update(**vit_kwargs)

        super(DinoVisionBackbone, self).__init__(
            init_values=init_values,
            ffn_layer="mlp",
            block_chunks=0,  # TODO: Get this from config. And it is important.
            num_register_tokens=0,
            interpolate_antialias=False,
            interpolate_offset=0.1,
            mlp_ratio=4,
            in_chans=in_channels,
            patch_size=patch_size,
            block_fn=partial(Block, attn_class=MemEffAttention),
            *args,
            **kwargs,
        )
        self.fpn_scaling = fpn_scaling
        self._is_init = False
        self.reuse_fpn_weights_from_reconstruction_head = reuse_fpn_weights_from_reconstruction_head

        # this is a hack to rescale the representations to the same resolution for all models
        self.rescale_representations = rescale_representations
        if self.rescale_representations:
            warnings.warn(
                "Rescaling representations is a hack. It is not recommended to use this option, only for fair comparison of patch_size 14/16 models."
            )

        if unwrap_checkpoint and (init_cfg is not None) and (not init_cfg["checkpoint"].endswith("unwrapped.pth")):
            checkpoint_path = init_cfg["checkpoint"]

            if self.reuse_fpn_weights_from_reconstruction_head:
                # Hack to use the FPN weights from the reconstruction head
                # i.e. use the pretraining reconstruction FPNs to initialize
                # the FPNs in the segmentaition head.
                def use_norm_in_fpn(state_dict):
                    assert "student.reconstruction_head.fpn1.0.weight" in state_dict["model"].keys()
                    assert "student.reconstruction_head.fpn1.0.weight" in state_dict["model"].keys()

                    layers = {1: [0, 1, 3], 2: [0]}

                    for i, ks in layers.items():
                        for k in ks:
                            state_dict["model"][f"teacher.backbone.fpn{i}.{k}.weight"] = copy.deepcopy(
                                state_dict["model"][f"student.reconstruction_head.fpn{i}.{k}.weight"]
                            )
                            state_dict["model"][f"teacher.backbone.fpn{i}.{k}.bias"] = copy.deepcopy(
                                state_dict["model"][f"student.reconstruction_head.fpn{i}.{k}.bias"]
                            )

                    state_dict["model"]["teacher.backbone.fpn1.1.running_mean"] = copy.deepcopy(
                        state_dict["model"]["student.reconstruction_head.fpn1.1.running_mean"]
                    )
                    state_dict["model"]["teacher.backbone.fpn1.1.running_var"] = copy.deepcopy(
                        state_dict["model"]["student.reconstruction_head.fpn1.1.running_var"]
                    )

                    return state_dict

                init_cfg["checkpoint"] = modify_checkpoint_keys_as_copy(
                    use_norm_in_fpn, checkpoint_path, "_reuse_fpn_weights"
                )

            checkpoint_path = init_cfg["checkpoint"]
            assert (
                "eval" not in checkpoint_path
            ), "We no longer finetune from the eval folder, but instead use prefix in the `init_cfg` to choose the teacher weights. Are you sure this is correct?"

            checkpoint = torch.load(checkpoint_path)
            # Extract directory, filename, and extension
            dir_name, file_name = os.path.split(checkpoint_path)
            base_name, ext = os.path.splitext(file_name)

            # Append "unwrapped" to the filename
            new_file_name = f"{base_name}_unwrapped{ext}"
            new_checkpoint_path = os.path.join(dir_name, new_file_name)

            torch.save(checkpoint["model"], new_checkpoint_path)
            init_cfg["checkpoint"] = new_checkpoint_path

        self.init_cfg = copy.deepcopy(init_cfg)

        if out_indices is None:
            self.out_index = self.default_out_indices[size]
        else:
            self.out_index = out_indices

        if self.fpn_scaling:
            self.fpn1, self.fpn2, self.fpn3, self.fpn4 = self.get_fpn_layers(fpn_scaling)

        BaseModule.init_weights(
            self
        )  # explicitly call BaseModule's init_weights as both parent classes have the same named fn

        if freeze_vit:
            for param in self.parameters():
                param.requires_grad = False
        else:
            for param in self.parameters():
                param.requires_grad = True

    def forward(self, x, *args, **kwargs):
        B, C, H, W = x.shape
        outputs = self.get_intermediate_layers(x=x, n=self.out_index, reshape=True, *args, **kwargs)

        if self.fpn_scaling:
            # FPN
            fpn1 = self.fpn1(outputs[0])
            fpn2 = self.fpn2(outputs[1])
            fpn3 = self.fpn3(outputs[2])
            fpn4 = self.fpn4(outputs[3])
            if self.rescale_representations:
                rescale_base = 14
                fpn1 = nn.functional.interpolate(fpn1, size=(rescale_base * 4, rescale_base * 4), mode="bilinear")
                fpn2 = nn.functional.interpolate(fpn2, size=(rescale_base * 2, rescale_base * 2), mode="bilinear")
                fpn3 = nn.functional.interpolate(fpn3, size=(rescale_base, rescale_base), mode="bilinear")
                fpn4 = nn.functional.interpolate(fpn4, size=(rescale_base // 2, rescale_base // 2), mode="bilinear")
            return [fpn1, fpn2, fpn3, fpn4]
        else:
            return outputs

    def get_fpn_layers(self, fpn_scaling):
        if fpn_scaling == 1:
            fpn1 = nn.Sequential(
                nn.ConvTranspose2d(self.embed_dim, self.embed_dim, kernel_size=2, stride=2),
                nn.SyncBatchNorm(self.embed_dim),
                nn.GELU(),
                nn.ConvTranspose2d(self.embed_dim, self.embed_dim, kernel_size=2, stride=2),
            )

            fpn2 = nn.Sequential(
                nn.ConvTranspose2d(self.embed_dim, self.embed_dim, kernel_size=2, stride=2),
            )

            fpn3 = nn.Identity()

            fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            raise NotImplementedError("Caution, only use fpn_scaling 1")

        return fpn1, fpn2, fpn3, fpn4


def modify_checkpoint_keys_as_copy(modifier_func, checkpoint_path, new_checkpoint_name_postfix):
    checkpoint = torch.load(checkpoint_path)
    assert checkpoint is not None

    # modiferier func: State_Dict -> State_Dict
    new_checkpoint = modifier_func(checkpoint)

    dir_name, file_name = os.path.split(checkpoint_path)
    base_name, ext = os.path.splitext(file_name)

    new_file_name = f"{base_name}_{new_checkpoint_name_postfix}{ext}"
    new_checkpoint_path = os.path.join(dir_name, new_file_name)

    torch.save(new_checkpoint, new_checkpoint_path)

    return new_checkpoint_path
