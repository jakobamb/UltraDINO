_base_ = [
    "../../../../_base_/datasets/placenta.py",
    "../../../../_base_/default_runtime.py",
    "../../../../_base_/schedules/schedule_20k.py",
    "../../../../_base_/models/dinus_segvit_vits16.py",
]

# Model settings
num_classes = 2
checkpoint_path = "/data/proto/Jakob/dinov2/out/vits_16_dinus_13m/eval/teacher_final_seg.pth"

# Model Definition
model = dict(
    backbone=dict(
        init_cfg=dict(type="Pretrained", checkpoint=checkpoint_path, prefix="backbone"),
    ),
    decode_head=dict(
        num_classes=num_classes,
        loss_decode=dict(type="ATMLoss", num_classes=num_classes, class_weights=(0.2, 0.8)),
    ),
)
