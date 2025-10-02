dataset_type = "FASSDataset"
data_root = "{{$DOWNSTREAM_DATA:/}}/fass/"

crop_size = (224, 224)

train_dataloader = dict(
    batch_size=64,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path="train/all/images", seg_map_path="train/all/annotations"),
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
        data_prefix=dict(img_path="val/images", seg_map_path="val/annotations"),
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
        data_prefix=dict(img_path="test/images", seg_map_path="test/annotations"),
    ),
)

val_evaluator = dict(type="DinusIoUMetric", iou_metrics=["mIoU", "mDice", "mFscore", "cmat"])
test_evaluator = val_evaluator
