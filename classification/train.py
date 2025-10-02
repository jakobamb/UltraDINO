#!/usr/bin/env python
import logging
import os
import hydra
from omegaconf import DictConfig
import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from dinov2.utils.utils import fix_random_seeds
from classification.models.lightning_model import ClassifierDINUS
from classification.data.classification_data import ClassificationDataModule

logger = logging.getLogger("CLS")


@hydra.main(config_path="./configs", config_name="fine_tune_sono41", version_base="1.1")
def train(cfg: DictConfig):
    base_dir = os.getcwd()
    logger.info("Current working directory (hydra run dir):", base_dir)
    logger.info("*** Configuration: ***")
    logger.info(cfg)

    # Make deterministic
    logger.warning("Using a random seed!")
    # fix_random_seeds(cfg.experiment.seed)

    # Number of GPUs
    num_devices = cfg.compute.devices

    data = ClassificationDataModule(
        dataset_name=cfg.datasets.dataset_name,
        img_size=(cfg.data.img_size, cfg.data.img_size),
        batch_size=cfg.training.batch_size,  # Adjusted batch size for each device
        num_workers=cfg.training.num_workers,
        in_chans=cfg.data.in_chans,
        normalization=cfg.data.normalization,
    )

    pt_weights = cfg.model.pretrained_weights
    # make sure that the base_dir is a subdirectory of the pretrained weights
    if not pt_weights:
        logger.warning("No pretrained weights specified. Training from scratch.")
    elif cfg.pt_conf and not pt_weights.startswith(cfg.pt_conf.train.output_dir):
        logger.warning(
            "CAUTION [USER WARNING]: Loading pretrained weights from a different directory than the pretraining output directory! \
                  Potentially, you need to adjust the output directory in the pretrain config, or you are loading the wrong weights."
        )

    log_dir = os.path.join(base_dir, "csv_logs")
    os.makedirs(log_dir, exist_ok=True)

    # Set up the CSVLogger without specifying the version
    csv_logger = CSVLogger(save_dir=log_dir, name=cfg.experiment.name)
    tb_logger = TensorBoardLogger(save_dir=os.path.join(base_dir, "tb_logs"), name=cfg.experiment.name)

    logger.info(f"Starting training {cfg.experiment.name} in dir {base_dir} over {cfg.training.epochs} epochs")

    # Set up the checkpoint callback using the logger's version
    if cfg.model.linear_probe:
        # only save the last model for linear probe
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(base_dir, "checkpoints", cfg.experiment.name),
            filename="last",
            save_last=True,
        )
    else:
        # finetuning: save only the best checkpoint based on validation accuracy
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(base_dir, "checkpoints", cfg.experiment.name),
            filename="best",
            save_top_k=1,  # Only save the best checkpoint
            save_last=True,
            save_on_train_epoch_end=True,
            monitor="val_acc",
            mode="min",
        )

    # Set up the trainer
    trainer = pl.Trainer(
        strategy="ddp" if num_devices > 1 else "auto",
        accelerator=cfg.compute.accelerator,
        devices=cfg.compute.devices,
        max_epochs=cfg.training.epochs,
        logger=[csv_logger, tb_logger],
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=cfg.training.check_val_every_n_epoch,
        overfit_batches=cfg.training.overfit_batches,  # Use overfitting as a debugging technique
        num_sanity_val_steps=0 if cfg.training.overfit_batches > 0 else 2,
    )

    # lp_learning_rates = cfg.training.lp_learning_rates
    # scale linear layer lr to batch size
    # lp_lrs_scaled = (
    #     lp_learning_rates * (cfg.training.batch_size) / 256.0
    # )  # TODO: Check if num_devices * bs = actual bs or if scaled. Also, this doesn't work.

    model = ClassifierDINUS(
        pt_config=cfg.pt_conf,
        backbone=cfg.model.backbone,
        pt_weights=cfg.model.pretrained_weights,
        num_classes=cfg.model.num_classes,
        imsize=cfg.data.img_size,
        linear_probe=cfg.model.linear_probe,
        layer_wise_lr_decay=cfg.model.layer_wise_lr_decay,
        weight_decay=cfg.optimizer.weight_decay,
        beta1=cfg.optimizer.beta1,
        in_chans=cfg.data.in_chans,
        beta2=cfg.optimizer.beta2,
        epochs=cfg.training.epochs,
        warmup_epochs=cfg.scheduler.warmup_epochs,
        batch_size_per_gpu=cfg.training.batch_size,
        label_smoothing=cfg.data.label_smoothing,
        mixup=cfg.data.mixup,
        cutmix=cfg.data.cutmix,
        learning_rate=cfg.training.learning_rate,
    )

    trainer.fit(model=model, datamodule=data)
    trainer.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


if __name__ == "__main__":
    train()
