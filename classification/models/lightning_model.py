import logging
import math
from functools import partial

import torch
import lightning.pytorch as pl
from torch.optim import AdamW
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC

from dinov2.eval.linear import create_linear_input, LinearClassifier, AllClassifiers
from dinov2.eval.utils import ModelWithIntermediateLayers
from dinov2.utils.param_groups import get_params_groups_with_decay, fuse_params_groups

from classification.models import backbones
from classification.utils.lr_scheduler import get_cosine_schedule_with_warmup

logger = logging.getLogger("CLS")


class ClassifierDINUS(pl.LightningModule):
    def __init__(
        self,
        pt_config: dict,
        num_classes: int,
        imsize: int,
        in_chans: int,
        pt_weights: str,
        backbone: str = "dino",
        linear_probe: bool = False,
        weight_decay: float = 0,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epochs: int = 100,
        learning_rate: float = 1e-3,
        label_smoothing: float = 0.1,
        mixup: float = 0.8,
        cutmix: float = 1.0,
        layer_wise_lr_decay: float = 0.65,
        warmup_epochs: int = 5,
        batch_size_per_gpu: int = 1024,
        lp_learning_rates: list[int] = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1],
        best_linear_classifier: str = None,
    ):
        super().__init__()

        # config
        self.pt_conf = pt_config
        self.num_classes = num_classes
        self.linear_probe = linear_probe

        # Loss, optimizer and scheduler parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.layer_wise_lr_decay = layer_wise_lr_decay
        self.batch_size_per_gpu = batch_size_per_gpu
        self.best_linear_classifier = best_linear_classifier

        if pt_config:
            # take patch_embed_lr_mult from dino config if available
            self.patch_embed_lr_mult = pt_config.optim.patch_embed_lr_mult
        else:
            # otherwise use reasonable default
            self.patch_embed_lr_mult = 0.2

        # Variables to be set in on_fit_start()
        self.num_gpus = None
        self.total_steps = None
        self.warmup_steps = None

        self.mixup_fn = None
        mixup_active = mixup > 0 or cutmix > 0
        if mixup_active:
            logger.info("Mixup is activated!")
            self.mixup_fn = Mixup(
                mixup_alpha=mixup,
                cutmix_alpha=cutmix,
                cutmix_minmax=None,
                prob=1.0,
                switch_prob=0.5,
                mode="batch",
                label_smoothing=label_smoothing,
                num_classes=num_classes,
            )

        if mixup_active:
            self.criterion = SoftTargetCrossEntropy()
        elif label_smoothing > 0:
            self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

        # Save params
        self.save_hyperparameters()

        # Model
        self.backbone_type = backbone
        if self.backbone_type == "dino":
            backbone_model, autocast_dtype = backbones.get_dinov2_backbone(
                pt_config, pt_weights=pt_weights, img_size=imsize
            )
        elif self.backbone_type == "usfm":
            backbone_model = backbones.get_usfm_backbone(pt_weights, img_size=imsize)
            autocast_dtype = torch.float16  # HACK
        elif self.backbone_type == "timm":
            logger.info("Using timm backbone with model string %s", pt_weights)
            backbone_model = backbones.get_timm_backbone(
                pt_weights,
                num_classes=num_classes,
            )
            autocast_dtype = torch.float16  # HACK
            self.model = backbone_model  # timm models already have classification head

        if linear_probe:
            assert self.backbone_type != "timm", "Linear probing not implemented for timm backbones."
            logger.info("Using linear probing -> freezing all layers but the head and initializing linear classifiers.")
            n_last_blocks_list = [1, 4]
            n_last_blocks = max(n_last_blocks_list)
            autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=autocast_dtype)
            self.model = ModelWithIntermediateLayers(backbone_model, n_last_blocks, autocast_ctx).cuda()
            sample_output = self.model(torch.rand(size=(1, in_chans, 224, 224), dtype=torch.float).cuda())
            self.linear_classifiers, self.lp_optim_param_groups = self._setup_linear_classifiers(
                sample_output, n_last_blocks_list, lp_learning_rates, num_classes
            )
            if self.best_linear_classifier is None:
                self.best_linear_classifier = ""
                self.best_lp_acc = 0
            self.lp_metrics = self._setup_lp_metrics(num_classes, self.linear_classifiers.classifiers_dict.keys())

            # freeze backbone
            for _, p in self.model.named_parameters():
                p.requires_grad = False
        else:
            if self.backbone_type != "timm":
                logger.info("Adding classification head to backbone.")
                # setup classification head
                batch_norm = torch.nn.BatchNorm1d(backbone_model.embed_dim, affine=False, eps=1e-6)
                cls_head = torch.nn.Linear(backbone_model.embed_dim, num_classes)
                self.model = torch.nn.Sequential(backbone_model, batch_norm, cls_head)

            logger.info("Finetuning entire model -> unfreezing all layers.")
            for _, p in self.model.named_parameters():
                p.requires_grad = True

        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.train_acc_per_class = Accuracy(task="multiclass", num_classes=num_classes, average=None)
        self.val_acc_per_class = Accuracy(task="multiclass", num_classes=num_classes, average=None)
        self.test_acc_per_class = Accuracy(task="multiclass", num_classes=num_classes, average=None)

        # Additional Metrics for Imbalanced Dataset
        self.train_precision = Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.train_recall = Recall(task="multiclass", num_classes=num_classes, average="macro")
        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.train_precision_per_class = Precision(task="multiclass", num_classes=num_classes, average=None)
        self.train_recall_per_class = Recall(task="multiclass", num_classes=num_classes, average=None)
        self.train_f1_per_class = F1Score(task="multiclass", num_classes=num_classes, average=None)

        self.val_precision = Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.val_recall = Recall(task="multiclass", num_classes=num_classes, average="macro")
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.val_precision_per_class = Precision(task="multiclass", num_classes=num_classes, average=None)
        self.val_recall_per_class = Recall(task="multiclass", num_classes=num_classes, average=None)
        self.val_f1_per_class = F1Score(task="multiclass", num_classes=num_classes, average=None)

        # additional metrics for test set
        self.test_precision = Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.test_recall = Recall(task="multiclass", num_classes=num_classes, average="macro")
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.test_precision_per_class = Precision(task="multiclass", num_classes=num_classes, average=None)
        self.test_recall_per_class = Recall(task="multiclass", num_classes=num_classes, average=None)
        self.test_f1_per_class = F1Score(task="multiclass", num_classes=num_classes, average=None)

        # AUROC Metric (requires probabilities)
        self.val_auroc = AUROC(task="multiclass", num_classes=num_classes, average="macro")
        self.test_auroc = AUROC(task="multiclass", num_classes=num_classes, average="macro")

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_orig = y.clone()
        if self.mixup_fn:
            x, y = self.mixup_fn(x, y)

        if self.linear_probe:
            features = self.model(x)
            outputs = self.linear_classifiers(features)
            losses = {f"loss_{k}": torch.nn.CrossEntropyLoss()(v, y) for k, v in outputs.items()}
            loss = sum(losses.values())
            for k, v in outputs.items():
                self.log(f"loss_{k}", losses[f"loss_{k}"], prog_bar=False, on_step=True, sync_dist=True)
                preds = torch.argmax(v, dim=1)
                self.lp_metrics[f"train_acc_{k}"](preds, y_orig)
                self.lp_metrics[f"train_acc_per_class_{k}"](preds, y_orig)
                self.log(f"train_acc_{k}", self.lp_metrics[f"train_acc_{k}"], prog_bar=False, sync_dist=True)
                self.log_multiclass_metrics(f"train_acc_per_class_{k}", self.lp_metrics[f"train_acc_per_class_{k}"])
        else:
            logits = self.model(x)
            loss = self.criterion(logits, y)
            # Get predictions
            preds = torch.argmax(logits, dim=1)

            # Update metrics with original labels
            self.train_acc(preds, y_orig)
            self.train_precision(preds, y_orig)
            self.train_recall(preds, y_orig)
            self.train_f1(preds, y_orig)
            self.train_acc_per_class(preds, y_orig)
            self.train_precision_per_class(preds, y_orig)
            self.train_recall_per_class(preds, y_orig)
            self.train_f1_per_class(preds, y_orig)

            self.log("train_loss", loss, prog_bar=True, on_step=True, sync_dist=True)
            self.log("train_acc", self.train_acc, prog_bar=True, on_step=True, sync_dist=True)
            self.log("train_precision", self.train_precision, prog_bar=True, on_step=True, sync_dist=True)
            self.log("train_recall", self.train_recall, prog_bar=True, on_step=True, sync_dist=True)
            self.log("train_f1", self.train_f1, prog_bar=True, on_step=True, sync_dist=True)

            self.log_multiclass_metrics("train_acc_per_class", self.train_acc_per_class)
            self.log_multiclass_metrics("train_precision_per_class", self.train_precision_per_class)
            self.log_multiclass_metrics("train_recall_per_class", self.train_recall_per_class)
            self.log_multiclass_metrics("train_f1_per_class", self.train_f1_per_class)

            # Log the learning rate
            lr = self.optimizers().param_groups[0]["lr"]
            last_layer_lr = self.optimizers().param_groups[-1]["lr"]
            self.log("learning_rate", lr, on_step=True, prog_bar=True, on_epoch=False, sync_dist=True)
            self.log("last_layer_lr", last_layer_lr, on_step=True, prog_bar=True, on_epoch=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        if self.linear_probe:
            features = self.model(x)
            outputs = self.linear_classifiers(features)
            losses = {f"loss_{k}": torch.nn.CrossEntropyLoss()(v, y) for k, v in outputs.items()}

            for k, v in outputs.items():
                self.log(f"loss_{k}", losses[f"loss_{k}"], prog_bar=False, on_step=True, sync_dist=True)
                preds = torch.argmax(v, dim=1)
                self.lp_metrics[f"val_acc_{k}"](preds, y)
                self.lp_metrics[f"val_acc_per_class_{k}"](preds, y)
                self.log(f"val_acc_{k}", self.lp_metrics[f"val_acc_{k}"], prog_bar=False, sync_dist=True)
                self.log_multiclass_metrics(f"val_acc_per_class_{k}", self.lp_metrics[f"val_acc_per_class_{k}"])

        else:
            self.log("val_acc", self.val_acc, prog_bar=True, sync_dist=True)
            self.log("val_precision", self.val_precision, prog_bar=True, sync_dist=True)
            self.log("val_recall", self.val_recall, prog_bar=True, sync_dist=True)
            self.log("val_f1", self.val_f1, prog_bar=True, sync_dist=True)
            self.log("val_auroc", self.val_auroc, prog_bar=True, sync_dist=True)

            self.log_multiclass_metrics("val_acc_per_class", self.val_acc_per_class)
            self.log_multiclass_metrics("val_precision_per_class", self.val_precision_per_class)
            self.log_multiclass_metrics("val_recall_per_class", self.val_recall_per_class)
            self.log_multiclass_metrics("val_f1_per_class", self.val_f1_per_class)

            logits = self.model(x)
            # Get predictions
            preds = torch.argmax(logits, dim=1)
            # Convert logits to probabilities for AUROC
            probs = torch.softmax(logits, dim=1)
            # Update metrics
            self.val_acc(preds, y)
            self.val_precision(preds, y)
            self.val_recall(preds, y)
            self.val_f1(preds, y)
            self.val_auroc(probs, y)

            self.val_acc_per_class(preds, y)
            self.val_precision_per_class(preds, y)
            self.val_recall_per_class(preds, y)
            self.val_f1_per_class(preds, y)

    def on_validation_end(self):
        if self.linear_probe:
            best_lp_acc = 0
            best_cls = ""

            for cls_name, v in self.linear_classifiers.classifiers_dict.items():
                acc = self.lp_metrics[f"val_acc_{cls_name}"].compute()
                if acc > best_lp_acc:
                    best_lp_acc = acc
                    best_cls = cls_name

            logger.info(f"best_linear_classifier: {best_cls}")
            logger.info(f"best_lp_acc: {best_lp_acc}")

        return super().on_validation_end()

    def test_step(self, batch, batch_idx):
        x, y = batch

        if self.linear_probe:
            features = self.model(x)
            outputs = self.linear_classifiers(features)
            losses = {f"loss_{k}": torch.nn.CrossEntropyLoss()(v, y) for k, v in outputs.items()}

            for k, v in outputs.items():
                self.log(f"loss_{k}", losses[f"loss_{k}"], prog_bar=True, on_step=True, sync_dist=True)
                preds = torch.argmax(v, dim=1)
                self.lp_metrics[f"test_acc_{k}"](preds, y)
                self.lp_metrics[f"test_f1_{k}"](preds, y)
                # self.lp_metrics[f"test_acc_per_class_{k}"](preds, y)
                self.log(f"test_acc_{k}", self.lp_metrics[f"test_acc_{k}"], prog_bar=True, sync_dist=True)
                self.log(f"test_f1_{k}", self.lp_metrics[f"test_f1_{k}"], prog_bar=True, sync_dist=True)
                # self.log_multiclass_metrics(f"test_acc_per_class_{k}", self.lp_metrics[f"test_acc_per_class_{k}"])

        else:
            # Get predictions
            logits = self.model(x)
            preds = torch.argmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1)
            # Update metrics
            self.test_acc(preds, y)
            self.test_precision(preds, y)
            self.test_recall(preds, y)
            self.test_f1(preds, y)
            self.test_auroc(probs, y)

            self.test_acc_per_class(preds, y)
            self.test_precision_per_class(preds, y)
            self.test_recall_per_class(preds, y)
            self.test_f1_per_class(preds, y)

            # Log metrics
            self.log("test_acc", self.test_acc, prog_bar=True, sync_dist=True)
            self.log("test_precision", self.test_precision, prog_bar=True, sync_dist=True)
            self.log("test_recall", self.test_recall, prog_bar=True, sync_dist=True)
            self.log("test_f1", self.test_f1, prog_bar=True, sync_dist=True)
            self.log("test_auroc", self.test_auroc, prog_bar=True, sync_dist=True)

            self.log_multiclass_metrics("test_acc_per_class", self.test_acc_per_class)
            self.log_multiclass_metrics("test_precision_per_class", self.test_precision_per_class)
            self.log_multiclass_metrics("test_recall_per_class", self.test_recall_per_class)
            self.log_multiclass_metrics("test_f1_per_class", self.test_f1_per_class)

    def get_maybe_fused_params_for_submodel(self, m, lr_decay_rate: float, path_embed_lr_mult: float):
        params_groups = get_params_groups_with_decay(
            model=m,
            lr_decay_rate=lr_decay_rate,
            patch_embed_lr_mult=path_embed_lr_mult,
        )
        fused_params_groups = fuse_params_groups(params_groups)
        logger.info("fusing param groups")

        for g in fused_params_groups:
            g["foreach"] = True
        return fused_params_groups

    def configure_optimizers(self):
        if self.linear_probe:
            logger.info("Setting up optimizer and scheduler for linear probing")
            optimizer = torch.optim.SGD(self.lp_optim_param_groups, momentum=0.9, weight_decay=0)

        else:
            # Collect all parameters
            logger.info("Setting up optimizer and scheduler for finetuning")
            logger.info("layer_wise_lr_decay: %f", self.layer_wise_lr_decay)
            logger.info("patch_embed_lr_mult: %f", self.patch_embed_lr_mult)

            all_param_groups = []
            # for m in self.model[0]._modules.values():

            if self.backbone_type == "timm":
                # timm vits
                all_param_groups += self.get_maybe_fused_params_for_submodel(
                    self.model, self.layer_wise_lr_decay, self.patch_embed_lr_mult
                )
            else:
                # dino or usfm
                all_param_groups += self.get_maybe_fused_params_for_submodel(
                    self.model[0], self.layer_wise_lr_decay, self.patch_embed_lr_mult
                )

                # add head to param groups
                all_param_groups += [
                    {"params": self.model[1].parameters(), "lr": self.learning_rate},  # BN
                    {"params": self.model[2].parameters(), "lr": self.learning_rate},  # Linear
                ]

            optimizer = AdamW(
                all_param_groups,
                lr=self.learning_rate,
                betas=(self.beta1, self.beta2),
                # weight_decay=self.weight_decay,  # XXX check if hparams make sense
            )

        # Ensure on_fit_start() has been called
        if self.total_steps is None or self.warmup_steps is None:
            self.on_fit_start()
            print("Total steps and warmup steps have been set.")
            print(f"Total steps: {self.total_steps}")
            print(f"Warmup steps: {self.warmup_steps}")

        # Configure scheduler with correct steps
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Scheduler steps every batch
                "frequency": 1,
            },
        }

    def on_fit_start(self):
        # This method is called when fit begins
        # Get number of GPUs (devices)
        self.num_gpus = max(1, self.trainer.num_devices)

        # Get the total number of samples
        train_dataloader = self.trainer.datamodule.train_dataloader()
        train_dataset = train_dataloader.dataset

        # If using a DistributedSampler, get the underlying dataset
        if isinstance(train_dataset, torch.utils.data.Subset):
            total_num_samples = len(train_dataset.dataset)
        else:
            total_num_samples = len(train_dataset)

        # Compute total batch size and steps per epoch
        total_batch_size = self.batch_size_per_gpu * self.num_gpus
        steps_per_epoch = math.ceil(total_num_samples / total_batch_size)

        # Calculate total steps and warmup steps
        self.total_steps = self.epochs * steps_per_epoch
        self.warmup_steps = self.warmup_epochs * steps_per_epoch

    def _setup_linear_classifiers(self, sample_output, n_last_blocks_list, learning_rates, num_classes=1000):
        """
        assuming lrs are already scaled to batch size
        """
        linear_classifiers_dict = torch.nn.ModuleDict()
        optim_param_groups = []
        for n in n_last_blocks_list:
            for avgpool in [False, True]:
                for lr in learning_rates:
                    out_dim = create_linear_input(sample_output, use_n_blocks=n, use_avgpool=avgpool).shape[1]
                    linear_classifier = LinearClassifier(
                        out_dim, use_n_blocks=n, use_avgpool=avgpool, num_classes=num_classes
                    )
                    linear_classifier = linear_classifier.cuda()
                    linear_classifiers_dict[
                        f"classifier_{n}_blocks_avgpool_{avgpool}_lr_{lr:.5f}".replace(".", "_")
                    ] = linear_classifier
                    optim_param_groups.append({"params": linear_classifier.parameters(), "lr": lr})

        linear_classifiers = AllClassifiers(linear_classifiers_dict)

        return linear_classifiers, optim_param_groups

    def _setup_lp_metrics(self, num_classes, keys):
        """
        Setup metrics for linear probing.

        Args:
            num_classes: Number of classes in the dataset.
            lrs: List of learning rates to setup metrics for.
        """
        lp_metrics = {}

        for classifier_name in keys:
            # convert lr to string in scientific notation
            lp_metrics[f"train_acc_{classifier_name}"] = Accuracy(task="multiclass", num_classes=num_classes)
            lp_metrics[f"train_acc_per_class_{classifier_name}"] = Accuracy(
                task="multiclass", num_classes=num_classes, average=None
            )
            lp_metrics[f"val_acc_{classifier_name}"] = Accuracy(task="multiclass", num_classes=num_classes)
            lp_metrics[f"val_acc_per_class_{classifier_name}"] = Accuracy(
                task="multiclass", num_classes=num_classes, average=None
            )
            lp_metrics[f"test_acc_{classifier_name}"] = Accuracy(task="multiclass", num_classes=num_classes)
            lp_metrics[f"test_f1_{classifier_name}"] = F1Score(
                task="multiclass", num_classes=num_classes, average="macro"
            )
            lp_metrics[f"test_acc_per_class_{classifier_name}"] = Accuracy(
                task="multiclass", num_classes=num_classes, average=None
            )

        # turn the dictionary into a TorchMetrics object
        lp_metrics = torch.nn.ModuleDict(lp_metrics)

        return lp_metrics

    def log_multiclass_metrics(self, metric_name, metric):
        """
        Logs multiclass metrics for each class (no averaging).

        Args:
            metric_name: Name of the metric.
            metric: Metric object to log.
        """

        class_labels = self.trainer.datamodule.class_labels
        class_metrics = metric.compute()

        if class_labels is None:
            # using indexes if no class_labels are provided
            class_labels = list(range(len(class_metrics)))

        assert len(class_metrics) == len(class_labels), "Number of class labels must match number of class metrics"
        assert len(class_metrics) == self.num_classes, "Number of class labels must match number of class metrics"

        for idx, metric_value in enumerate(class_metrics):  # Iterate over tensor elements.
            self.log(
                f"{metric_name}_{class_labels[idx]}",
                metric_value,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
