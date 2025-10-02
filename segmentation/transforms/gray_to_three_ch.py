from mmcv.transforms import TRANSFORMS
import numpy as np


@TRANSFORMS.register_module()
class GrayToThreeChannels:
    def __call__(self, results):
        if results["img"].ndim == 2:
            results["img"] = np.stack([results["img"], results["img"], results["img"]], axis=-1)
        return results
