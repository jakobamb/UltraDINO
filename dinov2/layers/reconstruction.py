# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock
from timm.models.vision_transformer import PatchEmbed, Block

from mmseg.models.decode_heads import UPerHead


class UperReconstrucionHeadLightweight(nn.Module):
    def __init__(self, embed_dim=768, out_channels=1):
        super().__init__()

        self.embed_dim = embed_dim

        # Feature Pyramid Network
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dim, self.embed_dim, kernel_size=2, stride=2),
            nn.SyncBatchNorm(self.embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(self.embed_dim, self.embed_dim, kernel_size=2, stride=2),
        )

        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dim, self.embed_dim, kernel_size=2, stride=2),
        )
        self.fpn3 = nn.Identity()
        self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # rec head
        self.rec_head = nn.Sequential(
            BasicBlock(self.embed_dim, self.embed_dim),
            nn.Conv2d(self.embed_dim, out_channels, kernel_size=1),
        )

    def forward_decoder(self, x):
        x1 = self.fpn1(x[0])
        x2 = self.fpn2(x[1])
        x3 = self.fpn3(x[2])
        x4 = self.fpn4(x[3])

        # rescale and add features
        x = x1
        x = x + F.interpolate(x2, size=x.shape[2:], mode="nearest")
        x = x + F.interpolate(x3, size=x.shape[2:], mode="nearest")
        x = x + F.interpolate(x4, size=x.shape[2:], mode="nearest")

        x = self.rec_head(x)

        return x

    def forward_loss(self, imgs, x):
        assert x.shape == imgs.shape
        loss = (x - imgs) ** 2
        loss = loss.mean()

        return loss

    def forward(self, imgs, x, rescale_img=True):
        x = self.forward_decoder(x)
        if rescale_img:
            B, C, H, W = x.shape
            imgs = F.interpolate(imgs, size=(H, W), mode="bilinear", align_corners=True)
        loss = self.forward_loss(imgs, x)
        return loss, x


class UperReconstructionHead(nn.Module):
    def __init__(self, embed_dim=768, pool_scales=(1, 2, 3, 6), out_channels=1, **kwargs):
        super().__init__()

        self.embed_dim = embed_dim

        # Feature Pyramid Network
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dim, self.embed_dim, kernel_size=2, stride=2),
            nn.SyncBatchNorm(self.embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(self.embed_dim, self.embed_dim, kernel_size=2, stride=2),
        )

        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dim, self.embed_dim, kernel_size=2, stride=2),
        )
        self.fpn3 = nn.Identity()
        self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # UPerHead
        self.uper_head = UPerHead(
            pool_scales=pool_scales,
            in_channels=(embed_dim,) * len(pool_scales),
            in_index=(0, 1, 2, 3),
            channels=embed_dim,
            num_classes=out_channels,
        )

    def forward_decoder(self, x):
        x1 = self.fpn1(x[0])
        x2 = self.fpn2(x[1])
        x3 = self.fpn3(x[2])
        x4 = self.fpn4(x[3])

        x = self.uper_head([x1, x2, x3, x4])
        return x

    def forward_loss(self, imgs, x):
        assert x.shape == imgs.shape
        loss = (x - imgs) ** 2
        loss = loss.mean()

        return loss

    def forward(self, imgs, x, rescale_img=True):
        x = self.forward_decoder(x)
        if rescale_img:
            B, C, H, W = x.shape
            imgs = F.interpolate(imgs, size=(H, W), mode="bilinear", align_corners=True)
        loss = self.forward_loss(imgs, x)
        return loss, x


class ReconstructionHead(nn.Module):
    def __init__(
        self,
        in_chans,
        patch_size,
        num_patches,
        decoder_depth=3,
        encoder_hidden_dim=2048,
        decoder_hidden_dim=256,
        decoder_num_heads=16,
        norm_pix_loss=False,
        p_norm=2,
        loss_on_masked_patches_only=True,
    ):
        super().__init__()
        self.patch_size = patch_size
        norm_layer = nn.LayerNorm
        self.in_chans = in_chans

        self.p_norm = p_norm

        self.decoder_embed = nn.Linear(encoder_hidden_dim, decoder_hidden_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_hidden_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, decoder_hidden_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_hidden_dim,
                    decoder_num_heads,
                    mlp_ratio=4,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_hidden_dim)
        self.decoder_pred = nn.Linear(decoder_hidden_dim, patch_size**2 * in_chans, bias=True)  # decoder to patch

        self.norm_pix_loss = norm_pix_loss

        self.loss_on_masked_patches_only = loss_on_masked_patches_only

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_decoder(self, x):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        # mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        # x = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        # x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        # x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        # x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 1, H, W]
        pred: [N, L, p*p*1]
        mask: [N, L], False is keep, True is masked.
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        if self.p_norm == 2:
            loss = (pred - target) ** 2
        elif self.p_norm == 1:
            loss = torch.abs(pred - target)
        else:
            raise NotImplementedError

        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        if self.loss_on_masked_patches_only:
            return (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        else:
            return loss.sum() / mask.sum()  # mean loss on all patches

    def forward(self, imgs, latent, mask):
        pred = self.forward_decoder(latent)  # [N, L, patch_size**2]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, self.unpatchify(pred)

    def patchify(self, imgs):
        """
        imgs: (N, 1, H, W)
        x: (N, L, patch_size**2)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * self.in_chans))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2)
        imgs: (N, 1, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p))
        x = torch.einsum("nhwpq->nhpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 1, h * p, h * p))
        return imgs
