dataset_type = "JNU_IFM_Dataset"
data_root = "{{$DOWNSTREAM_DATA:/}}"

crop_size = (224, 224)

train_dataloader = dict(
    batch_size=64,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path="jnu_ifm_split_4_train/train/images", seg_map_path="train/masks"),
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
        data_prefix=dict(img_path="jnu_ifm_split_4_train/val/images", seg_map_path="val/masks"),
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
        data_prefix=dict(img_path="jnu_ifm_split_4_train/test/images", seg_map_path="test/masks"),
    ),
)

val_evaluator = dict(type="DinusIoUMetric", iou_metrics=["mIoU", "mDice", "mFscore", "cmat"])
test_evaluator = val_evaluator
