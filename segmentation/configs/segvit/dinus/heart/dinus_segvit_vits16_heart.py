_base_ = [
    "../../../_base_/datasets/heart.py",
    "../../../_base_/default_runtime.py",
    "../../../_base_/schedules/schedule_45k.py",
    "../../../_base_/models/dinus_segvit_vits16.py",
]

# Model settings
num_classes = 26
checkpoint_path = "/data/proto/Jakob/dinov2/out/debug/model_final.rank_0.pth"

class_weights = (
    0.0201,
    0.2105,
    0.0091,
    0.0787,
    0.0048,
    0.0115,
    0.0364,
    0.0155,
    0.0811,
    0.0577,
    0.0938,
    0.0146,
    0.0556,
    0.0112,
    0.0218,
    0.0548,
    0.0315,
    0.0186,
    0.0238,
    0.0117,
    0.0226,
    0.0361,
    0.0093,
    0.0217,
    0.0361,
    0.0116,
)


# Model Definition
model = dict(
    backbone=dict(
        init_cfg=dict(type="Pretrained", checkpoint=checkpoint_path),
    ),
    decode_head=dict(
        num_classes=num_classes,
        loss_decode=dict(type="ATMLoss", num_classes=num_classes, class_weights=class_weights),
    ),
)
