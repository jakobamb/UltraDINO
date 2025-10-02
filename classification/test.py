#!/usr/bin/env python
import logging
import os
import hydra
from omegaconf import DictConfig
import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from dinov2.utils.utils import fix_random_seeds
from classification.models.lightning_model import ClassifierDINUS
from classification.data.classification_data import ClassificationDataModule

logger = logging.getLogger("CLS")


@hydra.main(config_path="./configs", config_name="fine_tune_sono41", version_base="1.1")
def test(cfg: DictConfig):
    # change the current working dir to the checkpoint dir
    # example: /dtu/p1/jakambs/dinus_experiments/vits16_from_scratch/20241126-000000-56542/classification/FetalPlanes/from_scratch_fetal_planes_barcelona/2024-12-16/13-08-36/checkpoints/from_scratch_fetal_planes_barcelona
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(cfg.test.checkpoint_path))))
    base_dir = os.getcwd()
    logger.info("Current working directory (hydra run dir):", base_dir)
    logger.info("*** Configuration: ***")
    logger.info(cfg)

    # Make deterministic
    fix_random_seeds(cfg.experiment.seed)

    # Number of GPUs
    num_devices = cfg.compute.devices

    # Set up the data module for testing
    data = ClassificationDataModule(
        dataset_name=cfg.datasets.dataset_name,
        img_size=(cfg.data.img_size, cfg.data.img_size),
        batch_size=cfg.test.batch_size,  # Adjusted batch size for testing
        num_workers=cfg.test.num_workers,
        in_chans=cfg.data.in_chans,
    )

    # Load the trained model from a checkpoint
    checkpoint_path = cfg.test.checkpoint_path
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")

    logger.info(f"Loading model checkpoint from {checkpoint_path}")

    # check if cfg.test.best_linear_classifier even exists
    if not hasattr(cfg.test, "best_linear_classifier"):
        best_linear_classifier = None
    else:
        best_linear_classifier = cfg.test.best_linear_classifier

    model = ClassifierDINUS.load_from_checkpoint(
        checkpoint_path,
        pt_config=cfg.pt_conf,
        backbone=cfg.model.backbone,
        pt_weights=cfg.model.pretrained_weights,
        num_classes=cfg.model.num_classes,
        in_chans=cfg.data.in_chans,
        imsize=cfg.data.img_size,
        linear_probe=cfg.model.linear_probe,
        layer_wise_lr_decay=cfg.model.layer_wise_lr_decay,
        weight_decay=cfg.optimizer.weight_decay,
        beta1=cfg.optimizer.beta1,
        beta2=cfg.optimizer.beta2,
        epochs=cfg.training.epochs,
        warmup_epochs=cfg.scheduler.warmup_epochs,
        batch_size_per_gpu=cfg.test.batch_size,
        label_smoothing=cfg.data.label_smoothing,
        mixup=cfg.data.mixup,
        cutmix=cfg.data.cutmix,
        learning_rate=cfg.training.learning_rate,
        strict=False,
        best_linear_classifier=best_linear_classifier,
    )

    # Set up logger for testing
    test_log_dir = os.path.join(base_dir, "test_logs")
    os.makedirs(test_log_dir, exist_ok=True)

    csv_logger = CSVLogger(save_dir=test_log_dir, name=cfg.experiment.name)
    tb_logger = TensorBoardLogger(save_dir=os.path.join(base_dir, "test_tb_logs"), name=cfg.experiment.name)

    logger.info(f"Starting testing {cfg.experiment.name} in dir {base_dir}")

    # Set up the trainer
    trainer = pl.Trainer(
        strategy="ddp" if num_devices > 1 else "auto",
        accelerator=cfg.compute.accelerator,
        devices=num_devices,
        logger=[csv_logger, tb_logger],
    )

    # Run testing
    test_results = trainer.test(model=model, datamodule=data)
    logger.info(f"Test results: {test_results}")

    if cfg.model.linear_probe:
        # test_acc_classifier_.* contains all the test accuracies. We only need the best one and print it
        best_test_acc = [
            v for k, v in test_results[0].items() if k.startswith(f"test_acc_{cfg.test.best_linear_classifier}")
        ][0]
        logger.info(f"Best LP test accuracy: {best_test_acc}")

        # same with F1
        best_test_f1 = [
            v for k, v in test_results[0].items() if k.startswith(f"test_f1_{cfg.test.best_linear_classifier}")
        ][0]
        logger.info(f"Best LP test F1: {best_test_f1}")

    trainer.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


if __name__ == "__main__":
    test()
