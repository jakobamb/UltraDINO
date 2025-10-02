# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset


@DATASETS.register_module()
class PlacentaDataset(BaseSegDataset):
    """Placente dataset.

    0 is for background, which
    IS included in 2 categories. Hence, ``reduce_zero_label`` is fixed to False.
    The ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '.png'.
    """

    METAINFO = dict(classes=("background", "placenta"), palette=[[120, 120, 120], [133, 19, 13]])  # Grey and red

    def __init__(self, img_suffix=".png", seg_map_suffix=".png", reduce_zero_label=False, **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, reduce_zero_label=reduce_zero_label, **kwargs
        )
