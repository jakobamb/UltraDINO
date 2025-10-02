model_name = "dinus_base_patch14_224"
img_size = (224, 224)
in_channels = 3
encoder_embed_dims = 768
out_indices = [5, 7, 11]
num_classes = 2

data_preprocessor = dict(type="SegDataPreProcessor", size=img_size)

# Model Definition
model = dict(
    type="EncoderDecoder",
    backbone=dict(
        type="DinoVisionBackbone",
        size="base",
        img_size=img_size,
        patch_size=14,
        freeze_vit=False,
        drop_path_rate=0.4,
        in_channels=in_channels,
        out_indices=out_indices,
        unwrap_checkpoint=False,
        # norm_cfg=norm_cfg,
        # out_indices=[8, 9, 10, 11]
    ),
    decode_head=dict(
        in_channels=encoder_embed_dims,
        channels=encoder_embed_dims,
        num_classes=num_classes,
    ),
    test_cfg=dict(mode="whole"),
)

find_unused_parameters = True
