from collections import defaultdict
from typing import Optional, TYPE_CHECKING, List
import copy
import os

import numpy as np
import torch
from torch import cat, Tensor, nn
from torch.nn.functional import avg_pool2d
from torch.utils.data import DataLoader

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

if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate


class MyPlugin(SupervisedPlugin):
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
        super().__init__()
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.batch_size_mem = batch_size_mem
        self.redius = 0
        if storage_policy is not None:  # Use other storage policy
            self.storage_policy = storage_policy
            assert storage_policy.max_size == self.mem_size
        else:  # Default
            self.storage_policy = ERBuffer(
                max_size=self.mem_size,
            )

    # def before_training_iteration(
    #         self, strategy: Template, *args, **kwargs
    # ) -> CallbackResult:
    #     # print(strategy.mbatch[0].shape)
    #
    #     strategy.mbatch[0] = torch.stack([torch.rot90(strategy.mbatch[0], k, (2, 3)) for k in range(4)], 1)
    #     strategy.mbatch[0] = strategy.mbatch[0].view(-1, 3, 32, 32)
    #     strategy.mbatch[1] = torch.stack([strategy.mbatch[1] for k in range(4)], 1).view(-1)

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
        strategy.buffer = self.storage_policy._buffer
        strategy.buffer_size = self.mem_size
        self.buffer1 = []
        self.buffer_label = []
        for i in strategy.buffer:
            for j in strategy.buffer[i]:
                self.buffer1.append(j)
                self.buffer_label.append(i)
        print('Current buffer size:', len(self.buffer1))
        strategy.exp_x = []
        strategy.exp_y = []
        print('Current:', len(self.buffer1))
        # if strategy.experience.current_experience == 10:
        #     strategy.train_epochs = 40
        # if strategy.experience.current_experience == 20:
        #     strategy.train_epochs = 30
        # if strategy.experience.current_experience == 40:
        #     strategy.train_epochs = 20

    def before_training_epoch(self, strategy: "SupervisedTemplate", **kwargs):
        self.index = list(range(self.mem_size))

    def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        # print("after_training_exp")
        m = []
        for i in strategy.exp_x:
            with torch.no_grad():
                k = strategy.model(i)[1]
                m.append(k)
        representation = torch.cat(tuple(m), dim=0)

        strategy.exp_y = torch.cat(tuple(strategy.exp_y), dim=0)
        # benchmark.seen_classes, benchmark.present_classes_in_each_exp
        # strategy.plugins[3].alpha =
        # 特征漂移
        # if strategy.experience.current_experience == 0:
        #     radius = []
        #     feature = np.array(representation.cpu())
        #     feature_dim = feature[1]
        #     labels = np.array(strategy.exp_y.cpu())
        #     y_set = np.unique(labels)
        #     print(y_set)
        #     for item in y_set:
        #         index = np.where(item == labels)[0]
        #         feature_classwise = feature[index]
        #         cov = np.cov(feature_classwise.T)
        #         radius.append(np.trace(cov) / feature_dim)
        #     print(radius)
        #     q = np.mean(radius)
        #     self.radius = np.sqrt(q)
        #     print(self.radius)
        # 动态参数
        b = strategy.benchmark
        total = len(b.seen_classes[strategy.experience.current_experience])
        p = b.first_occurrences
        dic = defaultdict(int)
        for i in p:
            dic[i] += 1
        if dic[strategy.experience.current_experience] != 0:
            strategy.plugins[3].alpha = min(50, total / dic[strategy.experience.current_experience] * 2)

        self.updatebuffer1(strategy, representation, strategy.exp_y)
        strategy.pre_exp_idx += 1

    def updatebuffer1(self, strategy: "SupervisedTemplate", re, y, **kwargs):
        self.storage_policy.update(strategy, re, y, **kwargs)

    def after_training_iteration(self, strategy, **kwargs):
        """
        Example callback: before backward.
        """
        if strategy.clock.train_exp_epochs == 0:
            strategy.exp_x.append(strategy.mb_x)
            strategy.exp_y.append(strategy.mb_y)

    def before_backward(
            self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:
        loss_buffer = self.compute_loss(strategy)
        # print(loss_buffer)
        strategy.loss += loss_buffer

    # @torch.no_grad()
    def compute_loss(self, strategy):
        proto_aug = []
        proto_aug_label = []
        np.random.shuffle(self.index)
        p = min(200, len(self.buffer1))
        for j in self.index[:p]:
            if j >= len(self.buffer1) or self.buffer1[j].shape != torch.Size([512]):
                # print(j)
                continue
            proto_aug.append(self.buffer1[j].reshape(1, 512))
            proto_aug_label.append(self.buffer_label[j])
        if proto_aug != []:
            # self.index = self.index[64:]
            proto_aug_label = torch.from_numpy(np.asarray(proto_aug_label)).to(strategy.device)
            proto_aug = torch.cat(tuple(proto_aug), dim=0)
            soft_feat = strategy.model.linear(proto_aug)
            loss_protoaug = nn.CrossEntropyLoss()(soft_feat, proto_aug_label)
            return loss_protoaug
        return 0
