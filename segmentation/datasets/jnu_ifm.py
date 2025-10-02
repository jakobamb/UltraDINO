# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset


@DATASETS.register_module()
class JNU_IFM_Dataset(BaseSegDataset):
    """
    JNU_IFM dataset.

    '.png'.
    """

    METAINFO = dict(
        classes=("background", "SP", "Head"),
        palette=[
            [120, 120, 120],  # grey background
            [133, 19, 13],  # red SP
            [17, 133, 13],  # green head
        ],
    )

    def __init__(self, img_suffix=".png", seg_map_suffix="_mask.png", reduce_zero_label=False, **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, reduce_zero_label=reduce_zero_label, **kwargs
        )
