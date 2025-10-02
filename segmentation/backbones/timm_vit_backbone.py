from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import timm
except ImportError:
    timm = None

import torch

from mmengine.model import BaseModule
from mmengine.registry import MODELS as MMENGINE_MODELS
from mmengine.logging import MMLogger
from mmseg.registry import MODELS


@MODELS.register_module()
class TIMMVisionTransformerBackbone(BaseModule):
    """Wrapper to use backbones from the timm library. More details can be found in
    `timm <https://github.com/rwightman/pytorch-image-models>`_.

    Args:
        model_name (str): Name of the timm model to instantiate.
        img_size (int or tuple): Input image size. Default: (512, 512).
        patch_size (int or tuple): Patch size for ViT models. Default: (16, 16).
        out_indices (int or list[int]): Indices of output features. Default: -1 (last layer).
        pretrained (bool): Load pretrained weights if True. Default: True.
        checkpoint_path (str): Path of a checkpoint to load after model initialization.
            Default: ''.
        in_channels (int): Number of input image channels. Default: 3.
        init_cfg (dict, optional): Initialization config dict. Default: None.
        **kwargs: Additional arguments for the timm model.
    """

    def __init__(
        self,
        model_name: str,
        img_size: Union[int, Tuple[int, int]] = (224, 224),
        patch_size: Union[int, Tuple[int, int]] = (16, 16),
        out_indices: Union[int, List[int]] = -1,
        pretrained: bool = False,
        checkpoint_path: str = "",
        in_channels: int = 1,
        freeze=False,
        init_cfg: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if timm is None:
            raise ImportError("The timm library is not installed.")
        super().__init__(init_cfg)

        # Resolve normalization layer if specified
        if "norm_layer" in kwargs:
            kwargs["norm_layer"] = MMENGINE_MODELS.get(kwargs["norm_layer"])

        # Create the timm model
        self.timm_model = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            in_chans=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            **kwargs,
        )

        # Check if the model supports 'get_intermediate_layers'
        if not hasattr(self.timm_model, "forward_intermediates"):
            raise AttributeError(f"The model '{model_name}' does not support 'get_intermediate_layers' method.")

        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

            # These are pytorch lightning checkpoints so we do this to get the actual model checkpoint
            checkpoint_model = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}

            # load pre-trained model
            # Here we use strict false since the checkpoint also carries the decoder weights.
            msg = self.timm_model.load_state_dict(checkpoint_model, strict=False)
            MMLogger.get_current_instance().info(msg)

            # The timm model has a head, that is the only allowed missing keys
            assert set(msg.missing_keys) == {"head.weight", "head.bias"}

        # Remove unnecessary attributes if they exist
        for attr in ["global_pool", "fc", "classifier", "head"]:
            if hasattr(self.timm_model, attr):
                setattr(self.timm_model, attr, None)

        # Freeze the vit if wanted
        if freeze:
            for param in self.timm_model.parameters():
                param.requires_grad = False

        # Hack to use pretrained or loaded weights from timm
        if pretrained or checkpoint_path:
            self._is_init = True

        # Determine the number of layers
        num_layers = self._get_num_layers()

        # Process out_indices
        self.out_indices = self._process_out_indices(out_indices, num_layers)

    def _get_num_layers(self) -> int:
        """Determine the number of layers in the model."""
        if hasattr(self.timm_model, "blocks"):
            return len(self.timm_model.blocks)
        elif hasattr(self.timm_model, "layers"):
            return sum(len(layer) for layer in self.timm_model.layers)
        else:
            raise AttributeError("Cannot determine the number of layers for this model.")

    def _process_out_indices(self, out_indices: Union[int, List[int]], num_layers: int) -> List[int]:
        """Process and validate out_indices."""
        if isinstance(out_indices, int):
            if out_indices == -1:
                out_indices = num_layers - 1
            indices = [out_indices]
        elif isinstance(out_indices, (list, tuple)):
            indices = list(out_indices)
        else:
            raise TypeError("out_indices must be an int, list, or tuple.")

        # Adjust negative indices and validate
        max_idx = num_layers - 1
        indices = [idx if idx >= 0 else num_layers + idx for idx in indices]
        for idx in indices:
            if idx < 0 or idx > max_idx:
                raise ValueError(f"Invalid layer index {idx}, must be between 0 and {max_idx}.")
        return indices

    def forward(self, x: Any) -> Tuple[Any, ...]:
        """Forward pass to extract features from specified layers.

        Args:
            x (Any): Input tensor.

        Returns:
            tuple: A tuple containing outputs from the specified layers.
        """
        outs = self.timm_model.forward_intermediates(
            x, indices=self.out_indices, norm=False, stop_early=False, output_fmt="NCHW", intermediates_only=True
        )
        return tuple(outs)
