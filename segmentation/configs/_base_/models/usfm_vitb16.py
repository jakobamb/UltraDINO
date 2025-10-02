img_size = (224, 224)
patch_size = 16
in_channels = 3
encoder_embed_dims = 768
out_indices = [3, 5, 7, 11]
num_classes = 2

data_preprocessor = dict(type="SegDataPreProcessor", size=img_size)

# Model Definition
model = dict(
    type="EncoderDecoder",
    backbone=dict(
        type="USFMViTBackbone",
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_channels,
        out_indices=out_indices,
        qkv_bias=True,
        drop_path_rate=0.1,
        use_abs_pos_emb=True,
        use_rel_pos_bias=False,
        use_shared_rel_pos_bias=False,
        pretrained=None,
    ),
    decode_head=dict(
        in_channels=encoder_embed_dims,
        channels=encoder_embed_dims,
        num_classes=num_classes,
    ),
    test_cfg=dict(mode="whole"),
)

find_unused_parameters = True
