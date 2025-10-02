from collections import defaultdict
from mmengine.optim import DefaultOptimWrapperConstructor
from mmengine.logging import print_log
from mmseg.registry import OPTIM_WRAPPER_CONSTRUCTORS


from functools import partial


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class USFMLearningRateDecayOptimizerConstructor(DefaultOptimWrapperConstructor):
    """Different learning rates are set for different layers of backbone."""

    def add_params(self, params, model, **kwargs):
        layer_decay = self.paramwise_cfg.get("layer_decay")
        weight_decay = self.paramwise_cfg.get("weight_decay")
        lr = self.paramwise_cfg.get("lr")
        param_groups = get_finetune_param_groups(model, lr=lr, layer_decay=layer_decay, weight_decay=weight_decay)
        params.extend(param_groups)


# Taken from USFM Github repository. All rights reserved.
def get_finetune_param_groups(model, lr, layer_decay, weight_decay, num_layers=12):
    parameter_group_names = {}
    parameter_group_vars = {}

    scales = list(layer_decay**i for i in reversed(range(num_layers + 2)))
    get_layer_func = partial(get_vit_layer, num_layers=num_layers + 2)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias"):
            group_name = "no_decay"
            this_weight_decay = 0.0
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_layer_func is not None:
            layer_id = get_layer_func(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if scales is not None:
                scale = scales[layer_id]
            else:
                scale = 1.0

            parameter_group_names[group_name] = {
                "group_name": group_name,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": lr * scale,
                "lr_scale": scale,
            }
            parameter_group_vars[group_name] = {
                "group_name": group_name,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": lr * scale,
                "lr_scale": scale,
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    return list(parameter_group_vars.values())


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


def get_vit_layer(name, num_layers):
    if name in ("cls_token", "mask_token", "pos_embed"):
        return 0
    elif name.startswith("patch_embed"):
        return 0
    elif name.startswith("rel_pos_bias"):
        return num_layers - 1
    elif name.startswith("blocks"):
        layer_id = int(name.split(".")[1])
        return layer_id + 1
    else:
        return num_layers - 1
