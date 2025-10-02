_base_ = [
    "../../../_base_/models/dinus_segvit_vits16.py",
    "../../../_base_/datasets/fass.py",
    "../../../_base_/pipelines/grayscale.py",
    "../../../_base_/default_runtime.py",
    "../../../_base_/schedules/schedule_45k.py",
]

num_classes = 5
checkpoint_path = "{{$EXPERIMENTS_DIR:/}}/vits16_dinus_13m/20241205-132822-468746/pretrain/eval/training_1999999/teacher_checkpoint.pth"

model = dict(
    type="EncoderDecoder",
    backbone=dict(
        freeze_vit=False,
        init_cfg=dict(type="Pretrained", checkpoint=checkpoint_path, prefix="backbone"),
    ),
    decode_head=dict(
        type="ATMHead",
        num_classes=5,
        loss_decode=dict(type="ATMLoss", num_classes=num_classes, class_weights=(1.0,) * num_classes),
    ),
)

train_dataloader = dict(dataset=dict(pipeline={{_base_.train_pipeline}}))
val_dataloader = dict(dataset=dict(pipeline={{_base_.test_pipeline}}))
test_dataloader = dict(dataset=dict(pipeline={{_base_.test_pipeline}}))
