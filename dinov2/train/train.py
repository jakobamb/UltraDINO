# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import argparse
import logging
import math
import os
from functools import partial

from fvcore.common.checkpoint import PeriodicCheckpointer
import torch

from dinov2.data import SamplerType, make_data_loader, make_dataset
from dinov2.data import (
    collate_data_and_cast,
    DataAugmentationDINO,
    DataAugmentationDINUS,
    MaskingGenerator,
    MaskingGeneratorMAE,
)
import dinov2.distributed as distributed
from dinov2.fsdp import FSDPCheckpointer
from dinov2.logging import MetricLogger, TBLogger
from dinov2.utils.config import setup
from dinov2.utils.utils import CosineScheduler
from dinov2.eval.rankme import rankMe
from dinov2.data.transforms import unnormalize

from dinov2.train.ssl_meta_arch import SSLMetaArch


torch.backends.cuda.matmul.allow_tf32 = True  # PyTorch 1.12 sets this to False by default
logger = logging.getLogger("dinov2")


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("DINOv2 training", add_help=add_help)
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Whether to not attempt to resume from the checkpoint directory. ",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--eval", type=str, default="", help="Eval type to perform")
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        type=str,
        help="Output directory to save logs and checkpoints",
    )

    return parser


def build_optimizer(cfg, params_groups):
    return torch.optim.AdamW(params_groups, betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2))


def build_schedulers(cfg):
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    lr = dict(
        base_value=cfg.optim["lr"],
        final_value=cfg.optim["min_lr"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.optim["warmup_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=0,
    )
    wd = dict(
        base_value=cfg.optim["weight_decay"],
        final_value=cfg.optim["weight_decay_end"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    momentum = dict(
        base_value=cfg.teacher["momentum_teacher"],
        final_value=cfg.teacher["final_momentum_teacher"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    teacher_temp = dict(
        base_value=cfg.teacher["teacher_temp"],
        final_value=cfg.teacher["teacher_temp"],
        total_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=cfg.teacher["warmup_teacher_temp"],
    )

    lr_schedule = CosineScheduler(**lr)
    wd_schedule = CosineScheduler(**wd)
    momentum_schedule = CosineScheduler(**momentum)
    teacher_temp_schedule = CosineScheduler(**teacher_temp)
    last_layer_lr_schedule = CosineScheduler(**lr)

    last_layer_lr_schedule.schedule[: cfg.optim["freeze_last_layer_epochs"] * OFFICIAL_EPOCH_LENGTH] = (
        0  # mimicking the original schedules
    )

    logger.info("Schedulers ready.")

    return (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    )


def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):
    for param_group in optimizer.param_groups:
        is_last_layer = param_group["is_last_layer"]
        lr_multiplier = param_group["lr_multiplier"]
        wd_multiplier = param_group["wd_multiplier"]
        param_group["weight_decay"] = wd * wd_multiplier
        param_group["lr"] = (last_layer_lr if is_last_layer else lr) * lr_multiplier


def do_test(cfg, model, iteration):
    new_state_dict = model.teacher.state_dict()

    if distributed.is_main_process():
        iterstring = str(iteration)
        eval_dir = os.path.join(cfg.train.output_dir, "eval", iterstring)
        os.makedirs(eval_dir, exist_ok=True)
        # save teacher checkpoint
        teacher_ckp_path = os.path.join(eval_dir, "teacher_checkpoint.pth")
        torch.save({"teacher": new_state_dict}, teacher_ckp_path)


def do_train(cfg, model, resume=False):
    model.train()
    inputs_dtype = torch.half
    fp16_scaler = model.fp16_scaler  # for mixed precision training

    # setup optimizer

    optimizer = build_optimizer(cfg, model.get_params_groups())
    (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    ) = build_schedulers(cfg)

    # checkpointer
    checkpointer = FSDPCheckpointer(model, cfg.train.output_dir, optimizer=optimizer, save_to_disk=True)

    start_iter = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1

    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    max_iter = cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH

    # Log gradient accumulation info
    accum_iter = getattr(cfg.train, "accum_iter", 1)
    if accum_iter > 1:
        logger.info(
            f"Using gradient accumulation with accum_iter={accum_iter}. "
            f"Note: This does NOT affect the effective batch size. "
            f"The number of iterations should be scaled accordingly."
        )
    effective_batch_size = cfg.train.batch_size_per_gpu * distributed.get_global_size()
    logger.info(f"Effective batch size: {effective_batch_size} (per step, across all processes)")

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer,
        period=3 * OFFICIAL_EPOCH_LENGTH,
        max_iter=max_iter,
        max_to_keep=3,
    )

    # setup data preprocessing

    img_size = cfg.crops.global_crops_size
    patch_size = cfg.student.patch_size
    n_tokens = (img_size // patch_size) ** 2
    if cfg.ibot.mask_style == "block":
        mask_generator = MaskingGenerator(
            input_size=(img_size // patch_size, img_size // patch_size),
            max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
        )
    elif cfg.ibot.mask_style == "patch":
        mask_generator = MaskingGeneratorMAE(
            input_size=(img_size // patch_size, img_size // patch_size),
        )

    # US specific transforms
    if "FUS" in cfg.train.dataset_path:
        data_transform = DataAugmentationDINUS(
            cfg.crops.global_crops_scale,
            cfg.crops.local_crops_scale,
            cfg.crops.local_crops_number,
            global_crops_size=cfg.crops.global_crops_size,
            local_crops_size=cfg.crops.local_crops_size,
        )
    else:
        data_transform = DataAugmentationDINO(
            cfg.crops.global_crops_scale,
            cfg.crops.local_crops_scale,
            cfg.crops.local_crops_number,
            global_crops_size=cfg.crops.global_crops_size,
            local_crops_size=cfg.crops.local_crops_size,
        )

    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_probability,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        dtype=inputs_dtype,
    )

    # setup data loader

    dataset = make_dataset(
        dataset_str=cfg.train.dataset_path,
        transform=data_transform,
        target_transform=lambda _: (),
    )
    # sampler_type = SamplerType.INFINITE
    sampler_type = SamplerType.SHARDED_INFINITE
    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        seed=start_iter,  # TODO: Fix this -- cfg.train.seed
        sampler_type=sampler_type,
        sampler_advance=0,  # TODO(qas): fix this -- start_iter * cfg.train.batch_size_per_gpu,
        drop_last=True,
        collate_fn=collate_fn,
    )

    # training loop

    iteration = start_iter

    logger.info("Starting training from iteration {}".format(start_iter))
    metrics_file = os.path.join(cfg.train.output_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    header = "Training"

    # tensorboard
    tb_writer = TBLogger(cfg.train.output_dir)
    tb_writer.log_config_to_hparams(cfg)

    rankme_embeddings_cls = []
    rankme_mean_embeddings_patches = []

    for data in metric_logger.log_every(
        data_loader,
        10,
        header,
        max_iter,
        start_iter,
    ):
        current_batch_size = data["collated_global_crops"].shape[0] / 2
        if iteration > max_iter:
            return

        # apply schedules

        lr = lr_schedule[iteration]
        wd = wd_schedule[iteration]
        mom = momentum_schedule[iteration]
        teacher_temp = teacher_temp_schedule[iteration]
        last_layer_lr = last_layer_lr_schedule[iteration]
        apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)

        # compute losses

        # Determine if we should perform optimizer step based on accumulation iterations
        accum_iter = getattr(cfg.train, "accum_iter", 1)
        should_update_grad = (iteration % accum_iter == 0) or (iteration == max_iter)

        # Only zero gradients when starting a new accumulation cycle
        if (iteration % accum_iter == 0) or (iteration == 1):
            optimizer.zero_grad(set_to_none=True)

        # Forward-backward pass
        loss_dict, additional_logs = model.forward_backward(data, teacher_temp=teacher_temp, iteration=iteration)

        # Only update weights after accumulating enough gradients
        if should_update_grad:
            # clip gradients
            if fp16_scaler is not None:
                if cfg.optim.clip_grad:
                    fp16_scaler.unscale_(optimizer)
                    for v in model.student.values():
                        v.clip_grad_norm_(cfg.optim.clip_grad)
                fp16_scaler.step(optimizer)
                fp16_scaler.update()
            else:
                if cfg.optim.clip_grad:
                    for v in model.student.values():
                        v.clip_grad_norm_(cfg.optim.clip_grad)
                optimizer.step()

        # perform teacher EMA update

        model.update_teacher(mom)

        # logging
        epoch = iteration // OFFICIAL_EPOCH_LENGTH

        # RankMe
        if cfg.rankme.enabled and distributed.is_main_process():
            assert current_batch_size * OFFICIAL_EPOCH_LENGTH >= cfg.rankme.num_samples

            iteration_in_epoch = iteration - (epoch * OFFICIAL_EPOCH_LENGTH)

            if (
                # should we do rankme this epoch?
                epoch % cfg.rankme.every_n_epochs == 0
                # only do rankme in first n samples of epoch
                and iteration_in_epoch * current_batch_size <= cfg.rankme.num_samples
            ):

                # (additional_logs["student_cls"] is batch_size x embed_dim
                # Note: we only append the first of the two global views
                rankme_embeddings_cls.append(additional_logs["student_backbone_cls"][: int(current_batch_size)])
                # (additional_logs["student_cls"] is batch_size x num_patches x embed_dim
                rankme_mean_embeddings_patches.append(
                    additional_logs["student_backbone_patches"][: int(current_batch_size)].mean(dim=1)
                )

                assert len(rankme_mean_embeddings_patches) == len(rankme_embeddings_cls)

                if len(rankme_embeddings_cls) * current_batch_size > cfg.rankme.num_samples:
                    z_cls = torch.cat(rankme_embeddings_cls, dim=0)
                    z_patch = torch.cat(rankme_mean_embeddings_patches, dim=0)

                    rankme_cls = rankMe(z_cls)
                    rankme_mean_patches = rankMe(z_patch)

                    rankme_embeddings_cls = []
                    rankme_mean_embeddings_patches = []

                    tb_writer.add_scalar("RankMe/CLS", rankme_cls, global_step=iteration)
                    tb_writer.add_scalar("RankMe/mPATCH", rankme_mean_patches, global_step=iteration)
                    tb_writer.add_scalar("RankMe/CLS num samples", z_cls.size(0), global_step=iteration)
                    tb_writer.add_scalar("RankMe/mPatch num samples", z_patch.size(0), global_step=iteration)

            else:
                # Make sure these are empty outside of the RankMe iterations
                rankme_embeddings_cls = []
                rankme_mean_embeddings_patches = []

        if distributed.get_global_size() > 1:
            for v in loss_dict.values():
                torch.distributed.all_reduce(v)
        loss_dict_reduced = {k: v.item() / distributed.get_global_size() for k, v in loss_dict.items()}

        if math.isnan(sum(loss_dict_reduced.values())):
            print("loss_dict", loss_dict)
            print("loss_dict_reduced", loss_dict_reduced)
            print("fp16_scaler", fp16_scaler)
            print("optim.clip_grad", cfg.optim.clip_grad)
            print("file_names", data["file_names"] if "file_names" in data.keys() else None)
            logger.info("NaN detected")
            raise AssertionError
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        metric_logger.update(lr=lr)
        metric_logger.update(wd=wd)
        metric_logger.update(mom=mom)
        metric_logger.update(last_layer_lr=last_layer_lr)
        metric_logger.update(current_batch_size=current_batch_size)
        metric_logger.update(total_loss=losses_reduced, **loss_dict_reduced)

        if iteration % cfg.train.tb_log_freq == 0:
            tb_writer.log_training_scalars(
                step=iteration,
                lr=lr,
                wd=wd,
                mom=mom,
                last_layer_lr=last_layer_lr,
                current_batch_size=current_batch_size,
            )
            tb_writer.log_losses(step=iteration, total_loss=losses_reduced, loss_dict=loss_dict_reduced)

        # checkpointing and testing
        if cfg.evaluation.eval_period_iterations > 0 and (iteration + 1) % cfg.evaluation.eval_period_iterations == 0:
            do_test(cfg, model, f"training_{iteration}")
            torch.cuda.synchronize()
        periodic_checkpointer.step(iteration)

        if iteration % (cfg.reconstruction.log_recs_every_n_epochs * OFFICIAL_EPOCH_LENGTH) == 0:
            if "reconstructions" in additional_logs:
                recs_un = unnormalize(additional_logs["reconstructions"])
                imgs_un = unnormalize(additional_logs["imgs"])
                tb_writer.add_images("plot/reconstructions", recs_un, global_step=iteration)
                tb_writer.add_images("plot/images", imgs_un, global_step=iteration)

        iteration = iteration + 1

    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args):
    cfg = setup(args)

    assert cfg.train.output_dir is not None

    model = SSLMetaArch(cfg).to(torch.device("cuda"))
    model.prepare_for_distributed_training()

    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        iteration = (
            FSDPCheckpointer(model, save_dir=cfg.train.output_dir)
            .resume_or_load(cfg.MODEL.WEIGHTS, resume=not args.no_resume)
            .get("iteration", -1)
            + 1
        )
        return do_test(cfg, model, f"manual_{iteration}")

    do_train(cfg, model, resume=not args.no_resume)


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)
