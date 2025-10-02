_base_ = [
    "../../_base_/models/segvit_b16.py",
    "../../_base_/datasets/placenta.py",
    "../../_base_/default_runtime.py",
    "../../_base_/schedules/schedule_45k.py",
]

model = dict(backbone=dict(model_name="vit_base_patch16_224.mae", pretrained=True))
