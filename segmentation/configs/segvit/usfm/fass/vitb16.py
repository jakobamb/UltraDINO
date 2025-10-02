_base_ = [
    "../../../_base_/models/usfm_vitb16.py",
    "../../../_base_/datasets/fass.py",
    "../../../_base_/pipelines/grayscale_3ch.py",
    "../../../_base_/default_runtime.py",
    "../../../_base_/schedules/schedule_45k.py",
]

num_classes = 5
checkpoint_path = "{{$EXPERIMENTS_DIR:/}}/vitb16_usfm/20241210-0000-0000/pretrain/usfm_latest_unwrapped.pth"
model_name = "usfm_upernet"

out_indices = [5, 7, 11]

model = dict(
    type="EncoderDecoder",
    backbone=dict(
        in_chans=3,
        type="USFMViTBackbone",
        qkv_bias=True,
        drop_path_rate=0.1,
        use_abs_pos_emb=False,
        use_rel_pos_bias=True,
        use_shared_rel_pos_bias=False,
        init_values=0.1,
        out_indices=out_indices,
        init_cfg=dict(type="Pretrained", checkpoint=checkpoint_path),
    ),
    decode_head=dict(
        type="ATMHead",
        img_size=(224, 224),
        in_channels=768,
        channels=768,
        num_layers=3,
        num_heads=12,
        use_stages=len(out_indices),
        embed_dims=384,
        num_classes=5,
        loss_decode=dict(
            type="ATMLoss", num_classes=num_classes, dec_layers=len(out_indices), class_weights=(1.0,) * num_classes
        ),
    ),
)

train_dataloader = dict(dataset=dict(pipeline={{_base_.train_pipeline}}))
val_dataloader = dict(dataset=dict(pipeline={{_base_.test_pipeline}}))
test_dataloader = dict(dataset=dict(pipeline={{_base_.test_pipeline}}))
