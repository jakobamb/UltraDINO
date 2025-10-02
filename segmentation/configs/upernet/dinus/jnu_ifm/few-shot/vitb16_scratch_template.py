_base_ = [
    "../../../../../_base_/models/dinus_vitb16.py",
    "../../../../../_base_/datasets/jnu_ifm.py",
    "../../../../../_base_/pipelines/grayscale.py",
    "../../../../../_base_/default_runtime.py",
    "../../../../../_base_/schedules/schedule_20k_v3.py",
]


num_classes = 5

split = [[SPLIT_INDEX]]

model = dict(
    type="EncoderDecoder",
    backbone=dict(
        type="DinoVisionBackbone",
        fpn_scaling=1,
        drop_path_rate=0.4,
        reuse_fpn_weights_from_reconstruction_head=False,
        freeze_vit=False,
        out_indices=[3, 5, 7, 11],
    ),
    decode_head=dict(
        type="UPerHead",
        channels=768,
        in_channels=[768, 768, 768, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=[1, 2, 3, 6],
        num_classes=5,
        loss_decode=[
            dict(type="CrossEntropyLoss", loss_name="loss_ce", use_sigmoid=False, loss_weight=1.0),
            dict(type="DiceLoss", loss_name="loss_dice", loss_weight=1.0),
        ],
    ),
)

train_dataloader = dict(
    dataset=dict(
        pipeline={{_base_.train_pipeline}},
        data_prefix=dict(
            img_path=f"jnu_ifm_splits/jnu_ifm_split_{split}/train/images",
            seg_map_path=f"jnu_ifm_splits/jnu_ifm_split_{split}/train/masks",
        ),
    )
)
val_dataloader = dict(
    dataset=dict(
        pipeline={{_base_.test_pipeline}},
        data_prefix=dict(
            img_path=f"jnu_ifm_splits/jnu_ifm_split_{split}/val/images",
            seg_map_path=f"jnu_ifm_splits/jnu_ifm_split_{split}/val/masks",
        ),
    )
)
test_dataloader = dict(
    dataset=dict(
        pipeline={{_base_.test_pipeline}},
        data_prefix=dict(
            img_path="jnu_ifm_splits/jnu_ifm_split_test/images", seg_map_path="jnu_ifm_splits/jnu_ifm_split_test/masks"
        ),
    )
)
