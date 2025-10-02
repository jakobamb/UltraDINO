import logging

from typing import Optional, Tuple

import torch
from torch import nn
from torch import dtype

from dinov2.eval.setup import get_autocast_dtype
from dinov2.models import build_model_from_cfg
import dinov2.utils.utils as dinov2_utils
from dinov2.models.vision_transformer import DinoVisionTransformer
from classification.models.usfm_vit import VisionTransformerUSFM

from urllib.request import urlopen
from PIL import Image
import timm

logger = logging.getLogger("CLS")


def get_dinov2_backbone(
    pt_conf: dict, img_size: int, pt_weights: Optional[str] = None
) -> Tuple[DinoVisionTransformer, dtype]:
    """
    Builds a DinoV2 from a pretraining config and loads weights, if available
    """
    print("pt_weights", pt_weights)
    pt_conf.crops.global_crops_size = img_size  # this is used to setup imsize/positional encodings
    model, _ = build_model_from_cfg(pt_conf, only_teacher=True)
    if pt_weights is not None:
        dinov2_utils.load_pretrained_weights(model, pt_weights, "teacher")
    else:
        logger.info("pt_weights is None, initializing model with random weights")

    autocast_dtype = get_autocast_dtype(pt_conf)
    return model, autocast_dtype


def get_usfm_backbone(pt_weights, **kwargs) -> VisionTransformerUSFM:
    # USFM config is the same as default ViT-B/16
    # model:
    #   model_name: vit
    #   model_type: FM
    #   resume: null
    #   model_cfg:
    #     type: vit
    #     num_classes: ${data.num_classes}
    #     backbone:
    #       pretrained: null
    #     name: vit-b
    #     in_chans: 3
    #     patch_size: 16
    #     embed_dim: 768
    #     depth: 12
    #     num_heads: 12
    #     mlp_ratio: 4
    #     qkv_bias: true
    #     attn_drop_rate: 0.0
    #     drop_path_rate: 0.1
    #     init_values: 0.1
    #     use_abs_pos_emb: false
    #     use_rel_pos_bias: true
    #     use_shared_rel_pos_bias: false
    #     use_mean_pooling: true

    # train:
    #   label_smoothing: 0.1

    vit = VisionTransformerUSFM(
        qkv_bias=True,
        drop_path_rate=0.1,
        # init_values=0.1,
        # # use_abs_pos_emb=False,    # Not sure about these. They are not in the state dict
        # use_rel_pos_bias=True,
        use_shared_rel_pos_bias=True,
        use_mean_pooling=True,
    )
    if pt_weights is not None:
        load_state_dict_non_strict(vit, pt_weights)
    else:
        logger.info("pt_weights is None, initializing model with random weights.")
        logger.info("Make sure that you are training from scratch or that the finetuning weights will be loaded later.")
    vit.forward = (
        vit.forward_features
    )  # overwrite forward method to return features instead of head for use in nn.sequential

    return vit


def get_timm_backbone(model_str, num_classes, pretrained=True):
    model = timm.create_model(model_str, pretrained=pretrained, num_classes=num_classes)
    return model


def load_state_dict_non_strict(model, state_dict_path):
    """
    Loads a PyTorch model state dict in a non-strict way and logs missing or unexpected keys.

    Args:
        model (torch.nn.Module): The model to load the state dict into.
        state_dict_path (str): Path to the state dictionary file.

    Returns:
        None
    """
    # Load the state dict
    state_dict = torch.load(state_dict_path)

    # Load state dict into the model with strict=False
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    # Log missing keys
    if missing_keys:
        logger.warning(f"Missing keys in the state dict: {missing_keys}")

    # Log unexpected keys
    if unexpected_keys:
        logger.warning(f"Unexpected keys in the state dict: {unexpected_keys}")
