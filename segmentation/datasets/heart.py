# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset


@DATASETS.register_module()
class HeartDataset(BaseSegDataset):
    """Placenta dataset.

    0 is for background, which we do not included in 29 categories.
    Hence, ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to '.png'.
    """

    METAINFO = dict(
        classes=(
            "aortic arch",
            "aortic valve",
            "ascending aorta",
            "brachiocephalic artery",
            "descending aorta",
            "ductus arteriosus",
            "inferior vena cava",
            "left atrium",
            "left common carotid artery",
            "left pulmonary artery",
            "left subclavian artery",
            "left ventricle",
            "left ventricular outflow tract",
            "main pulmonary artery",
            "mitral valve",
            "pulmonary valve",
            "pulmonary veins",
            "right atrium",
            "right pulmonary artery",
            "right ventricle",
            "septum primum",
            "stomach bubble",
            "superior vena cava",
            "tricuspid valve",
            "umbilical vein",
            "ventricular septum",
        ),
        palette=[
            # [120, 120, 120], # background - grey
            [133, 19, 13],  # aortic arch - dark red
            [120, 32, 80],  # aortic valve - burgundy
            [210, 105, 30],  # ascending aorta - orange-brown
            [123, 104, 238],  # brachiocephalic artery - light purple
            [72, 61, 139],  # descending aorta - dark slate blue
            [30, 144, 255],  # ductus arteriosus - dodger blue
            [135, 206, 235],  # inferior vena cava - sky blue
            # [218, 165, 32],  # ithmus - golden rod
            # [34, 139, 34],   # kidney - forest green
            [178, 34, 34],  # left atrium - firebrick red
            [255, 215, 0],  # left common carotid artery - gold
            [255, 140, 0],  # left pulmonary artery - dark orange
            [107, 142, 35],  # left subclavian artery - olive drab
            [255, 99, 71],  # left ventricle - tomato
            [165, 42, 42],  # left ventricular outflow tract - brown
            [240, 230, 140],  # main pulmonary artery - khaki
            [85, 107, 47],  # mitral valve - dark olive green
            # [128, 0, 128],   # portal sinus - purple
            [147, 112, 219],  # pulmonary valve - medium purple
            [0, 128, 128],  # pulmonary veins - teal
            [255, 0, 0],  # right atrium - red
            [70, 130, 180],  # right pulmonary artery - steel blue
            [0, 255, 0],  # right ventricle - lime green
            [128, 128, 0],  # septum primum - olive
            [210, 180, 140],  # stomach bubble - tan
            [0, 255, 255],  # superior vena cava - cyan
            [255, 105, 180],  # tricuspid valve - hot pink
            [112, 128, 144],  # umbilical vein - slate gray
            [245, 222, 179],  # ventricular septum - wheat
        ],
    )

    def __init__(self, img_suffix=".png", seg_map_suffix=".png", reduce_zero_label=True, **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, reduce_zero_label=reduce_zero_label, **kwargs
        )
