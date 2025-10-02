import copy
import inspect
from typing import List, Optional

import mmengine
from mmcv.transforms import BaseTransform
from mmcv.transforms import TRANSFORMS

try:
    import albumentations
    from albumentations import Compose

    ALBU_INSTALLED = True
except ImportError:
    albumentations = None
    Compose = None
    ALBU_INSTALLED = False


@TRANSFORMS.register_module()
class AlbuGray(BaseTransform):
    """Albumentation augmentation for grayscale images.

    This transform does not perform any RGB conversions.

    Args:
        transforms (list[dict]): A list of Albumentations transformations.
        keymap (dict): Mapping from original keys to Albumentations-style keys.
        additional_targets (dict, optional): Apply same augmentations to multiple
            objects of the same type.
        update_pad_shape (bool): Whether to update padding shape according to
            the output shape of the last transform.
    """

    def __init__(
        self,
        transforms: List[dict],
        keymap: Optional[dict] = None,
        additional_targets: Optional[dict] = None,
        update_pad_shape: bool = False,
    ):
        if not ALBU_INSTALLED:
            raise ImportError(
                "Albumentations is not installed. Please install it by running "
                '"pip install albumentations>=0.3.2 --no-binary qudida,albumentations"'
            )

        # Copy transforms to avoid modifying the original list
        transforms = copy.deepcopy(transforms)

        self.transforms = transforms
        self.keymap = keymap
        self.additional_targets = additional_targets
        self.update_pad_shape = update_pad_shape

        self.aug = Compose([self.albu_builder(t) for t in self.transforms], additional_targets=self.additional_targets)

        if not keymap:
            self.keymap_to_albu = {"img": "image", "gt_seg_map": "mask"}
        else:
            self.keymap_to_albu = copy.deepcopy(keymap)
        self.keymap_back = {v: k for k, v in self.keymap_to_albu.items()}

    def albu_builder(self, cfg: dict) -> object:
        """Builds a callable Albumentations transform from a config dict.

        Args:
            cfg (dict): Config dict containing the transform type and parameters.

        Returns:
            object: Albumentations transform object.
        """
        assert isinstance(cfg, dict) and "type" in cfg, "Config must be a dict containing the key 'type'."

        args = cfg.copy()
        obj_type = args.pop("type")

        if mmengine.is_str(obj_type):
            obj_cls = getattr(albumentations, obj_type)
        elif inspect.isclass(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(f"'type' must be a valid type or str, but got {type(obj_type)}")

        if "transforms" in args:
            args["transforms"] = [self.albu_builder(t) for t in args["transforms"]]

        return obj_cls(**args)

    @staticmethod
    def mapper(d: dict, keymap: dict) -> dict:
        """Maps the keys of a dictionary according to the provided keymap.

        Args:
            d (dict): Original dictionary.
            keymap (dict): Mapping from old keys to new keys.

        Returns:
            dict: Dictionary with updated keys.
        """
        return {keymap.get(k, k): v for k, v in d.items()}

    def transform(self, results: dict) -> dict:
        """Applies the Albumentations transforms to the results dict.

        Args:
            results (dict): Dictionary containing the data to transform.

        Returns:
            dict: Transformed results.
        """
        # Map keys to Albumentations format
        results = self.mapper(results, self.keymap_to_albu)

        # Apply Albumentations transforms
        results = self.aug(**results)

        # Map keys back to original format
        results = self.mapper(results, self.keymap_back)

        # Update padding shape if required
        if self.update_pad_shape:
            results["pad_shape"] = results["img"].shape

        return results

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(transforms={self.transforms})"
