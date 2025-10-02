# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from datetime import datetime
import logging
import os

# Hack to see if importing torch before mm stuff works
from dotenv import load_dotenv

import torch

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner

from mmseg.engine.optimizers import LayerDecayOptimizerConstructor

from mmseg.registry import RUNNERS


from segmentation.backbones import timm_vit_backbone, vit_dinov2, vit_usfm
from segmentation.datasets import fass, placenta, heart, jnu_ifm
from segmentation.decode_heads import atm_head, uper_upscale
from segmentation.losses import atm_loss
from segmentation.metrics.iou import DinusIoUMetric
from segmentation.transforms import albu_gray, gray_to_three_ch
from segmentation.hooks.mlflow import AsbjornsMlflowLoggerHook
from segmentation.optim.dinov2_layer_decay_optimizer_constructor import DinoV2LearningRateDecayOptimizerConstructor
from segmentation.optim.usfm_layer_decay_optimizer_constructor import USFMLearningRateDecayOptimizerConstructor


def parse_args():
    parser = argparse.ArgumentParser(description="Train a segmentor")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work-dir", help="overwrite default dir to save logs and models")
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="resume from the latest checkpoint in the work_dir automatically",
    )
    parser.add_argument("--amp", action="store_true", default=False, help="enable automatic-mixed-precision training")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument("--launcher", choices=["none", "pytorch", "slurm", "mpi"], default="none", help="job launcher")
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def main():
    assert torch.cuda.is_available(), "CUDA is not available"

    load_dotenv()
    args = parse_args()

    # load config

    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > config
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    else:
        now = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        head = cfg.model.decode_head.type
        dataset = cfg.dataset_type

        # pretrained model!
        if "checkpoint_path" in cfg.keys() is not None:
            remove_from = cfg.checkpoint_path.find("pretrain")
            assert (
                remove_from != -1
            ), "checkpoint path didnt contain 'pretrain'. Are you sure you are doing everything right?"

            parent_dir = os.path.abspath(cfg.checkpoint_path[:remove_from])

            if "dinus" in parent_dir:
                assert "eval" not in parent_dir, "This is most def wrong!"
            assert "pretrain" not in parent_dir, "This is really wrong."

        # from scratch!
        else:
            shorten_model_name = {
                "vit_small_patch16_224": "vits16",
                "vit_base_patch16_224": "vitb16",
                "dinus_small_patch16_224": "dinus_vits16",
                "dinus_base_patch16_224": "dinus_vitb16",
            }
            model = f"{shorten_model_name.get(cfg.model_name, cfg.model_name)}_scratch"
            parent_dir = os.path.join(os.environ["EXPERIMENTS_DIR"], model, "0000-0000")

        config_name, _ = os.path.splitext(os.path.basename(args.config))
        cfg.work_dir = os.path.join(parent_dir, "segmentation", dataset, head, config_name, now)

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == "AmpOptimWrapper":
            print_log("AMP training is already enabled in your config.", logger="current", level=logging.WARNING)
        else:
            assert optim_wrapper == "OptimWrapper", (
                "`--amp` is only supported when the optimizer wrapper type is "
                f"`OptimWrapper` but got {optim_wrapper}."
            )
            cfg.optim_wrapper.type = "AmpOptimWrapper"
            cfg.optim_wrapper.loss_scale = "dynamic"

    # resume training
    cfg.resume = args.resume

    # build the runner from config
    if "runner_type" not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()


if __name__ == "__main__":
    main()
