# optimizer
optimizer = dict(type="AdamW", lr=0.00001, betas=(0.9, 0.999), weight_decay=0.01)
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=optimizer,
    paramwise_cfg=dict(
        custom_keys={
            "norm": dict(decay_mult=0.0),
            "ln": dict(decay_mult=0.0),
            "head": dict(lr_mult=10.0),
        }
    ),
    clip_grad=None,
)

# learning policy for 20000 steps
param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=1e-6,
        by_epoch=False,
        begin=0,
        end=1000,
    ),
    dict(
        type="PolyLR",
        eta_min=0.0,
        power=1.0,
        by_epoch=False,
        begin=1500,
        end=20000,
    ),
]
# training schedule for 45000 steps
train_cfg = dict(type="IterBasedTrainLoop", max_iters=20000, val_interval=450)  # Validate 100 times
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=100, log_metric_by_epoch=False),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=False,
        interval=-1,  # Disable periodic checkpoint saving
        save_optimizer=True,
        save_last=True,  # Do not save the latest checkpoint
        max_keep_ckpts=1,  # Keep only the best checkpoint
        save_best="mDice",  # Save the best checkpoint based on mIoU
        rule="greater",  # Higher mIoU indicates better performance
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook", draw=True),
)
