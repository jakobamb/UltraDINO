# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.dist import master_only
from mmengine.hooks import LoggerHook
from mmseg.registry import HOOKS


@HOOKS.register_module()
class AsbjornsMlflowLoggerHook(LoggerHook):

    def __init__(
        self, exp_name=None, tags=None, log_model=True, interval=10, ignore_last=True, reset_flag=False, by_epoch=True
    ):
        super().__init__(interval, ignore_last, reset_flag, by_epoch)
        self.import_mlflow()
        self.exp_name = exp_name
        self.tags = tags
        self.log_model = log_model

    def import_mlflow(self):
        try:
            import mlflow
            import mlflow.pytorch as mlflow_pytorch
        except ImportError:
            raise ImportError('Please run "pip install mlflow" to install mlflow')
        self.mlflow = mlflow
        self.mlflow_pytorch = mlflow_pytorch

        import pdb

        pdb.set_trace()

    @master_only
    def before_run(self, runner):
        super().before_run(runner)
        if self.exp_name is not None:
            self.mlflow.set_experiment(self.exp_name)
        if self.tags is not None:
            self.mlflow.set_tags(self.tags)

    @master_only
    def log(self, runner):
        tags = self.get_loggable_tags(runner)
        if tags:
            self.mlflow.log_metrics(tags, step=self.get_iter(runner))

    @master_only
    def after_run(self, runner):
        if self.log_model:
            self.mlflow_pytorch.log_model(runner.model, "models")
