_base_ = [
    '../../_base_/datasets/placenta.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_45k.py'
]

# Model settings
model_name = 'dinus_small_patch16_224'
img_size = (224, 224)
patch_size = 16
in_channels = 1
encoder_embed_dims = 384
out_indices = [5, 7, 11]
num_classes = 2

checkpoint_path = '/data/proto/Jakob/dinov2/out/debug/model_final.rank_0.pth'

# Data Preprocessor
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[0.0],  # Hack to normalize images to [0,1], from preprocessor: inputs = [(_input - self.mean) / self.std for _input in inputs]
    std=[255.0],
    pad_val=0,
    seg_pad_val=255,
    bgr_to_rgb=False,
    rgb_to_bgr=False,
    size=img_size
)

# Model Definition
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='DinoVisionBackbone',
        size='base',
        img_size=img_size,
        patch_size=patch_size,
        freeze_vit=True,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_path),
        in_channels=in_channels,
        # norm_cfg=norm_cfg,
        # out_indices=[8, 9, 10, 11]
    ),
    decode_head=dict(
        type='BNHead',
        in_channels=([encoder_embed_dims]*4),
        in_index=[0, 1, 2, 3],
        input_transform='resize_concat',
        channels=3072,
        dropout_ratio=0,
        num_classes=21,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    )
    test_cfg=dict(mode='whole'),
)

find_unused_parameters=True

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    # dict(type='WandbVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=vis_backends)