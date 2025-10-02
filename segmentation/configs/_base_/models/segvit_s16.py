# Model settings
model_name = "vit_small_patch16_224"
img_size = (224, 224)
patch_size = (16, 16)
in_channels = 3
encoder_embed_dims = 768
out_indices = [5, 7, 11]
num_classes = 5

# Data Preprocessor
data_preprocessor = dict(
    type="SegDataPreProcessor",
    mean=[
        0.0
    ],  # Hack to normalize images to [0,1], from preprocessor: inputs = [(_input - self.mean) / self.std for _input in inputs]
    std=[255.0],
    pad_val=0,
    seg_pad_val=255,
    bgr_to_rgb=False,
    rgb_to_bgr=False,
    size=img_size,
)

# Model Definition
model = dict(
    type="EncoderDecoder",
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type="TIMMVisionTransformerBackbone",
        model_name=model_name,
        img_size=img_size,
        patch_size=patch_size,
        out_indices=out_indices,
        pretrained=False,
        checkpoint_path="",
        in_channels=in_channels,
        init_cfg=None,
    ),
    decode_head=dict(
        type="ATMHead",
        img_size=img_size,
        in_channels=encoder_embed_dims,
        channels=encoder_embed_dims,
        num_classes=num_classes,
        num_layers=4,
        num_heads=12,
        use_stages=len(out_indices),
        embed_dims=encoder_embed_dims // 2,
        loss_decode=dict(
            type="ATMLoss",
            num_classes=num_classes,
            dec_layers=len(out_indices),
            loss_weight=1.0,
            class_weights=(1.0, 1.0, 1.0, 1.0, 1.0),
        ),
    ),
    test_cfg=dict(mode="whole"),
)

find_unused_parameters = True
