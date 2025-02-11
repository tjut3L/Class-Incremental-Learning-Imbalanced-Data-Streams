
from random import random
from typing import Optional, TYPE_CHECKING, List
import copy
import os
from timm.optim import create_optimizer
import numpy as np
import torch
from avalanche.core import Template, CallbackResult
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate
import torch.nn.functional as F

class VPTPlugin(SupervisedPlugin):
    """
    Implemented your plugin (if any) here.
    """

    def __init__(
            self,
            mem_size: int = 200,
            batch_size: int = None,
            batch_size_mem: int = None,
            task_balanced_dataloader: bool = False,
            storage_policy: Optional["ERBuffer"] = None,
    ):


        self.cur_exp = 0
        self.n_classes = 100

        super().__init__()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # main training / eval actions here


    def before_training_iteration(
            self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:
        pass


    def before_backward(
            self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:
        strategy.loss += self.loss(strategy.mb_output,strategy.mb_y)
        pass
    def before_training_epoch(
        self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:
        strategy.model.train()
        pass


    def freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False
        return model

    def un_freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = True
        return model

    def before_training_exp(
            self,
            strategy: "SupervisedTemplate",
            num_workers: int = 0,
            shuffle: bool = True,
            **kwargs
    ):
        """
        Dataloader to build batches containing examples from both memories and
        the training dataset
        """
        strategy.model.eval()

    def loss(self, logits, targets):
        loss = F.cross_entropy(logits, targets, reduction="none")

        return torch.sum(loss) / targets.shape[0]


    def after_trining_exp(self, strategy: "SupervisedTemplate", **kwargs):
       return

    def after_training_epoch(
        self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:
        strategy.scheduler.step()

        # Enable eval mode
        strategy.model.eval()
        # acc1, acc5 = accuracy(strategy.mb_output, strategy.mb_y, topk=(1, 5))
        # print("ACC@1:",acc1)
        # print("ACC@5:",acc5)
        pass

    def after_training_iteration(self, strategy, **kwargs):
        """
        Example callback: before backward.
        """
        pass


    def after_eval_iteration(
            self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:
       pass

    def after_eval_exp(
            self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:
        pass

