_base_ = [
    "../../../_base_/models/dinus_segvit_vits16.py",
    "../../../_base_/datasets/fass.py",
    "../../../_base_/pipelines/grayscale.py",
    "../../../_base_/default_runtime.py",
    "../../../_base_/schedules/schedule_45k.py",
]

num_classes = 5

model = dict(
    type="EncoderDecoder",
    backbone=dict(freeze_vit=False),
    decode_head=dict(
        type="ATMHead",
        num_classes=5,
        loss_decode=dict(type="ATMLoss", num_classes=num_classes, class_weights=(1.0,) * num_classes),
    ),
)

train_dataloader = dict(dataset=dict(pipeline={{_base_.train_pipeline}}))
val_dataloader = dict(dataset=dict(pipeline={{_base_.test_pipeline}}))
test_dataloader = dict(dataset=dict(pipeline={{_base_.test_pipeline}}))
