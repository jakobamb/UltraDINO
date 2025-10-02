# dataset settings
dataset_type = "PlacentaDataset"
data_root = "/data/proto/jonas_line/datasets/segmentation_placenta"
train_pipeline = [
    dict(type="LoadImageFromFile", color_type="grayscale"),
    dict(type="LoadAnnotations", reduce_zero_label=False),
    dict(type="Resize", scale=(224, 224), keep_ratio=False),
    dict(type="RandomFlip", prob=[0.5, 0.5], direction=["horizontal", "vertical"]),
    dict(type="RandomRotate", prob=0.5, degree=90),
    dict(type="PackSegInputs"),
]
test_pipeline = [
    dict(type="LoadImageFromFile", color_type="grayscale"),
    dict(type="Resize", scale=(224, 224), keep_ratio=False),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to resize data transform
    dict(type="LoadAnnotations", reduce_zero_label=False),
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
        data_prefix=dict(img_path="images_no_empty/val", seg_map_path="annotations_no_empty/val"),
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
        data_prefix=dict(img_path="images_no_empty/test", seg_map_path="annotations_no_empty/test"),
        pipeline=test_pipeline,
    ),
)

val_evaluator = dict(type="IoUMetric", iou_metrics=["mIoU", "mDice", "mFscore"])
test_evaluator = val_evaluator
