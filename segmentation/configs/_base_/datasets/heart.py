# dataset settings
dataset_type = "HeartDataset"
data_root = "/data/proto/jonas_line/datasets/segmentation_heart"
train_pipeline = [
    dict(type="LoadImageFromFile", color_type="grayscale"),
    dict(type="LoadAnnotations", reduce_zero_label=True),
    dict(type="Resize", scale=(224, 224), keep_ratio=False),
    dict(
        type="AlbuGray",
        transforms=[
            dict(type="RandomBrightnessContrast", brightness_limit=[-0.3, 0.3], contrast_limit=[-0.3, 0.3], p=0.5),
            dict(type="RandomGamma", gamma_limit=(80, 120), p=0.5),
            dict(type="GaussNoise", var_limit=(10.0, 50.0), p=0.5),
            dict(type="GridDistortion", distort_limit=(-0.3, 0.3), p=0.5),
            dict(
                type="Affine",
                scale=(0.8, 1.2),
                translate_percent=(0.2, 0.2),
                rotate=(-30, 30),
                shear=(-15, 15),
                interpolation=1,
                mode=1,
                keep_ratio=True,
                p=0.5,
            ),
            dict(type="HorizontalFlip", p=0.5),
        ],
        keymap={"img": "image", "gt_seg_map": "mask"},
        update_pad_shape=False,
    ),
    dict(type="PackSegInputs"),
]
test_pipeline = [
    dict(type="LoadImageFromFile", color_type="grayscale"),
    dict(type="Resize", scale=(224, 224), keep_ratio=False),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to resize data transform
    dict(type="LoadAnnotations", reduce_zero_label=True),
    dict(type="PackSegInputs"),
]
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path="images/train", seg_map_path="annotations/train"),
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path="images/val", seg_map_path="annotations/val"),
        pipeline=test_pipeline,
    ),
)
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path="images/test", seg_map_path="annotations/test"),
        pipeline=test_pipeline,
    ),
)

val_evaluator = dict(type="IoUMetric", iou_metrics=["mIoU", "mDice", "mFscore"])
test_evaluator = val_evaluator
