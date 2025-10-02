from collections import defaultdict
from mmengine.optim import DefaultOptimWrapperConstructor
from mmengine.logging import print_log
from mmseg.registry import OPTIM_WRAPPER_CONSTRUCTORS


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class DinoV2LearningRateDecayOptimizerConstructor(DefaultOptimWrapperConstructor):
    """Different learning rates are set for different layers of backbone."""

    def add_params(self, params, model, **kwargs):
        decay_rate = self.paramwise_cfg.get("decay_rate")
        param_groups = get_params_groups_with_decay(model, lr_decay_rate=decay_rate)
        params.extend(param_groups)


# Taken from DINOV2 Github repository. All rights reserved.
def get_vit_lr_decay_rate(name, lr_decay_rate=1.0, num_layers=12, force_is_backbone=False, chunked_blocks=False):
    """
    Calculate lr decay rate for different ViT blocks.
    Args:
        name (string): parameter name.
        lr_decay_rate (float): base lr decay rate.
        num_layers (int): number of ViT blocks.
    Returns:
        lr decay rate for the given parameter.
    """
    layer_id = num_layers + 1
    if name.startswith("backbone") or force_is_backbone:
        if (
            ".pos_embed" in name
            or ".patch_embed" in name
            or ".mask_token" in name
            or ".cls_token" in name
            or ".register_tokens" in name
        ):
            layer_id = 0
        elif force_is_backbone and (
            "pos_embed" in name
            or "patch_embed" in name
            or "mask_token" in name
            or "cls_token" in name
            or "register_tokens" in name
        ):
            layer_id = 0
        elif ".blocks." in name and ".residual." not in name:
            layer_id = int(name[name.find(".blocks.") :].split(".")[2]) + 1
        elif chunked_blocks and "blocks." in name and "residual." not in name:
            layer_id = int(name[name.find("blocks.") :].split(".")[2]) + 1
        elif "blocks." in name and "residual." not in name:
            layer_id = int(name[name.find("blocks.") :].split(".")[1]) + 1

    return lr_decay_rate ** (num_layers + 1 - layer_id)


def fuse_params_groups(all_params_groups, keys=("lr_multiplier", "wd_multiplier", "is_last_layer")):
    fused_params_groups = defaultdict(lambda: {"params": []})
    for d in all_params_groups:
        identifier = ""
        for k in keys:
            identifier += k + str(d[k]) + "_"

        for k in keys:
            fused_params_groups[identifier][k] = d[k]
        fused_params_groups[identifier]["params"].append(d["params"])

    return fused_params_groups.values()


def get_params_groups_with_decay(model, lr_decay_rate=1.0, patch_embed_lr_mult=1.0):
    chunked_blocks = False
    if hasattr(model, "n_blocks"):
        print_log("chunked fsdp")
        n_blocks = model.n_blocks
        chunked_blocks = model.chunked_blocks
    elif hasattr(model, "blocks"):
        print_log("first code branch")
        n_blocks = len(model.blocks)
    elif hasattr(model, "backbone"):
        print_log("second code branch")
        n_blocks = len(model.backbone.blocks)
    else:
        print_log("else code branch")
        n_blocks = 0
    all_param_groups = []

    for name, param in model.named_parameters():
        name = name.replace("_fsdp_wrapped_module.", "")
        if not param.requires_grad:
            continue
        decay_rate = get_vit_lr_decay_rate(
            name, lr_decay_rate, num_layers=n_blocks, force_is_backbone=n_blocks > 0, chunked_blocks=chunked_blocks
        )
        d = {"params": param, "is_last_layer": False, "lr_multiplier": decay_rate, "wd_multiplier": 1.0, "name": name}

        if "last_layer" in name:
            d.update({"is_last_layer": True})

        if name.endswith(".bias") or "norm" in name or "gamma" in name:
            d.update({"wd_multiplier": 0.0})

        if "patch_embed" in name:
            d.update({"lr_multiplier": d["lr_multiplier"] * patch_embed_lr_mult})

        all_param_groups.append(d)
        print_log(f"""{name}: lr_multiplier: {d["lr_multiplier"]}, wd_multiplier: {d["wd_multiplier"]}""")

    return all_param_groups
