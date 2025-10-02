_base_ = [
    "../../../../../_base_/models/ibot_vits16.py",
    "../../../../../_base_/datasets/fass.py",
    "../../../../../_base_/pipelines/grayscale_3ch.py",
    "../../../../../_base_/default_runtime.py",
    "../../../../../_base_/schedules/schedule_45k_v3.py",
]

num_classes = 5
checkpoint_path = "{{$EXPERIMENTS_DIR:/}}/vits16_ibot_im1k/20250220-0000/pretrain/ibot_vits_im1k.pth"

subset = "[[SUBSET]]"
split = [[SPLIT_INDEX]]

model = dict(
    type="EncoderDecoder",
    backbone=dict(
        type="DinoVisionBackbone",
        fpn_scaling=1,
        init_values=None,
        reuse_fpn_weights_from_reconstruction_head=False,
        freeze_vit=False,
        out_indices=[3, 5, 7, 11],
        init_cfg=dict(type="Pretrained", checkpoint=checkpoint_path),
    ),
    decode_head=dict(
        type="UPerHead",
        channels=384,
        in_channels=[384, 384, 384, 384],
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
        data_prefix=dict(img_path=f"train/{subset}/{split}/images", seg_map_path=f"train/{subset}/{split}/annotations"),
    )
)

val_dataloader = dict(dataset=dict(pipeline={{_base_.test_pipeline}}))
test_dataloader = dict(dataset=dict(pipeline={{_base_.test_pipeline}}))
