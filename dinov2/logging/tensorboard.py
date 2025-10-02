import os

from collections.abc import MutableMapping
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


def get_timestamp():
    now = datetime.now()
    # total seconds since midnight
    total_seconds = now.hour * 3600 + now.minute * 60 + now.second
    # Format the timestamp as year_month_day_seconds
    timestamp = now.strftime(f"%Y_%m_%d_{total_seconds}")

    return timestamp


class TBLogger(SummaryWriter):
    def __init__(self, logdir):
        super().__init__(os.path.join(logdir, get_timestamp()))

    def log_training_scalars(self, step, lr, wd, mom, last_layer_lr, current_batch_size):
        self.add_scalar("learning rate", lr, global_step=step)
        self.add_scalar("weight decay", wd, global_step=step)
        self.add_scalar("momentum", mom, global_step=step)
        self.add_scalar("last layer learning rate", last_layer_lr, global_step=step)
        self.add_scalar("current batch size", current_batch_size, global_step=step)

    def log_losses(self, step, total_loss, loss_dict):
        self.add_scalar("total loss", total_loss, global_step=step)

        for k, v in loss_dict.items():
            self.add_scalar(f"{k}", v, global_step=step)

    def _flatten_dict(self, d, parent_key="", sep="."):
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, MutableMapping):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def _parse_config_to_hparams(self, config):
        """Convert config dictionary to a format suitable for TensorBoard hparams."""
        # Flatten the config dictionary
        flattened_config = self._flatten_dict(config)

        # Convert each key-value pair into the correct type for hparams
        hparams = {}
        for key, value in flattened_config.items():
            if isinstance(value, (int, float, str, bool)):
                hparams[key] = value
            else:
                # Convert complex types (like lists) to strings
                hparams[key] = str(value)

        return hparams

    def log_config_to_hparams(self, config, log_dir="out/debug"):
        """Log a config dictionary as hparams to TensorBoard."""
        # Parse config into hparams
        hparams = self._parse_config_to_hparams(config)
        # Log hparams to TensorBoard
        self.add_hparams(hparams, {})
