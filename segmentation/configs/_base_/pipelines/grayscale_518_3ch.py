train_pipeline = [
    dict(type="LoadImageFromFile", color_type="grayscale"),
    dict(type="LoadAnnotations"),
    dict(
        type="AlbuGray",
        transforms=[
            dict(
                type="Affine",
                scale=(0.8, 1.2),
                translate_percent=(0.2, 0.2),
                rotate=(-30, 30),
                shear=(-15, 15),
                cval=0,
                mode=0,  # cv2.BORDER_CONSTANT
                interpolation=1,
                keep_ratio=True,
                p=0.5,
            ),
            dict(type="HorizontalFlip", p=0.5),
            dict(type="RandomBrightnessContrast", brightness_limit=[-0.3, 0.3], contrast_limit=[-0.3, 0.3], p=0.5),
            dict(type="RandomGamma", gamma_limit=(80, 120), p=0.5),
            dict(type="GaussNoise", var_limit=(10.0, 50.0), p=0.5),
            dict(
                type="GridDistortion", distort_limit=(-0.3, 0.3), p=0.5, border_mode=0, value=0
            ),  # border_mode is cv2.BORDER_CONSTANT
        ],
        keymap={"img": "image", "gt_seg_map": "mask"},
        update_pad_shape=False,
    ),
    dict(type="GrayToThreeChannels"),
    dict(
        type="Normalize", mean=(0.485 * 255, 0.456 * 255, 0.406 * 255), std=(0.229 * 255, 0.224 * 255, 0.225 * 255)
    ),  # MEAN and STD from imagenet
    dict(type="Resize", scale=(518, 518)),
    dict(type="PackSegInputs"),
]
test_pipeline = [
    dict(type="LoadImageFromFile", color_type="grayscale"),
    dict(type="LoadAnnotations"),
    dict(type="GrayToThreeChannels"),
    dict(
        type="Normalize", mean=(0.485 * 255, 0.456 * 255, 0.406 * 255), std=(0.229 * 255, 0.224 * 255, 0.225 * 255)
    ),  # MEAN and STD from imagenet
    dict(type="Resize", scale=(518, 518)),
    dict(type="PackSegInputs"),
]
