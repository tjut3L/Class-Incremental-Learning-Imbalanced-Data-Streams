import pickle
from collections import defaultdict
from typing import Optional, TYPE_CHECKING, List
import copy
import os
from timm.optim import create_optimizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from timm.utils import accuracy
from torch import cat, Tensor, nn
from torch.nn.functional import avg_pool2d
from torch.utils.data import DataLoader
from copy import deepcopy

from torchvision import transforms
import torch.nn.functional as F
from avalanche.benchmarks.utils import concat_classification_datasets, AvalancheDataset
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.core import Template, CallbackResult
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.storage_policy import (
    ExemplarsBuffer,
    ExperienceBalancedBuffer, HerdingSelectionStrategy, ERBuffer
)
import math
from avalanche.models import FeatureExtractorBackbone
from avalanche.benchmarks.utils.classification_dataset import \
    classification_subset
from sklearn.metrics.pairwise import euclidean_distances
if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate


class CodaPlugin(SupervisedPlugin):
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
        self.valid_out_dim = 0
        self.last_valid_out_dim = 0
        self.cur_exp = 0
        self.criterion_fn = nn.CrossEntropyLoss(reduction='none')
        self.n_classes = 100
        super().__init__()

    def before_training_iteration(
            self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:
        pass


    def before_backward(
            self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:
        loss, output = self.update_model(strategy,strategy.mb_x, strategy.mb_y)
        strategy.loss += loss
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
        self.cur_exp = strategy.experience.current_experience
        if self.cur_exp > 0:
            try:
                if strategy.model.module.prompt is not None:
                    strategy.model.module.prompt.process_task_count()
            except:
                if strategy.model.prompt is not None:
                    strategy.model.prompt.process_task_count()
        self.add_valid_output_dim(strategy)
        self.data_weighting()
    def after_trining_exp(self, strategy: "SupervisedTemplate", **kwargs):
       return

    def after_training_epoch(
        self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:
        # acc1, acc5 = accuracy(strategy.mb_output, strategy.mb_y, topk=(1, 5))
        # print("ACC@1:",acc1)
        # print("ACC@5:",acc5)
        pass
    def after_backward(
        self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:
        best_model = deepcopy(strategy.model.state_dict())
        strategy.model.load_state_dict(best_model)
        # EXEMPLAR MANAGEMENT -- select training subset
        self.last_valid_out_dim = self.valid_out_dim
        strategy.optimizer.step()
    def after_training_iteration(self, strategy, **kwargs):
        """
        Example callback: before backward.
        """
        pass

    @torch.enable_grad()
    def update_model(self, strategy, inputs, targets):

        # logits
        logits, prompt_loss = strategy.model(inputs, train=True)
        logits = logits[:, :self.valid_out_dim]

        # ce with heuristic

        m = []
        k = strategy.benchmark.present_classes_in_each_exp[self.cur_exp].numpy().tolist()
        for i in range(100):
            if i not in k:
                m.append(i)
        # logits[:,m] = -float('inf')
        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        dw_cls = dw_cls.to(strategy.device)
        total_loss = self.criterion(logits, targets.long(), dw_cls)
        prompt_loss = prompt_loss.to(strategy.device)
        # ce loss
        total_loss = total_loss + prompt_loss.sum()


        return total_loss, logits

    def after_eval_iteration(
            self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:
       pass

    def after_eval_exp(
            self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:
        pass

    def criterion(self, logits, targets, data_weights):
        loss_supervised = (self.criterion_fn(logits, targets.long()) * data_weights).mean()
        return loss_supervised
    def forward(self, x):
        return self.model.forward(x)[:, :self.valid_out_dim]

    def data_weighting(self):
        self.dw_k = torch.tensor(np.ones(self.valid_out_dim + 1, dtype=np.float32))
        # cuda
        if self.cuda:
            self.dw_k = self.dw_k.cuda()

    def cuda(self,stratrgy):
        # torch.cuda.set_device(self.device)
        self.model = stratrgy.model.to(stratrgy.device)
        self.criterion_fn = self.criterion_fn.to(stratrgy.device)
        # Multi-GPU
        return self

    def add_valid_output_dim(self, strategy):
        # This function is kind of ad-hoc, but it is the simplest way to support incremental class learning
        print('Incremental class: Old valid output dimension:', self.valid_out_dim)
        self.valid_out_dim = 100
        print('Incremental class: New Valid output dimension:', self.valid_out_dim)
        return self.valid_out_dim


from torch.optim import Optimizer
import math


class _LRScheduler(object):
    def __init__(self, optimizer, last_epoch=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.
        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class CosineSchedule(_LRScheduler):

    def __init__(self, optimizer, K):
        self.K = K
        super().__init__(optimizer, -1)

    def cosine(self, base_lr):
        return base_lr * math.cos((99 * math.pi * (self.last_epoch)) / (200 * (self.K - 1)))

    def get_lr(self):
        return [self.cosine(base_lr) for base_lr in self.base_lrs]
