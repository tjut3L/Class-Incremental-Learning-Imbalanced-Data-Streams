from collections import defaultdict
from typing import Optional, TYPE_CHECKING, List
import copy
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
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


class DKDPlugin(SupervisedPlugin):
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
        self.exp_y = None
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.batch_size_mem = batch_size_mem
        self.redius = 0
        self.protos = []
        self.proto_labels = []
        self.old_model = None
        self.new_S = {}
        self.eval_count = 0
        self.flag = False
        if storage_policy is not None:  # Use other storage policy
            self.storage_policy = storage_policy
            assert storage_policy.max_size == self.mem_size
        else:  # Default
            self.storage_policy = ERBuffer(
                max_size=self.mem_size,
            )
    def freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False
        return model

    def un_freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = True
        return model
    def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        """保存模型并冻结"""
        self.old_model = deepcopy(strategy.model)
        self.old_model = self.freeze_model(self.old_model)


    def before_backward(
            self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:
        # X_mix, y, y_shuffle, lam = self.mixup_data(strategy, strategy.mb_x, strategy.mb_y)
        # # 将混合后的数据送入模型进行训练
        # out = strategy.model(torch.Tensor.float(X_mix))[1]
        # loss_mix = lam * torch.nn.CrossEntropyLoss()(out, y) + (1 - lam) * torch.nn.CrossEntropyLoss()(out, y_shuffle)
        # strategy.loss += loss_mix
        # strategy.loss += self.compute_loss(strategy)
        if strategy.experience.current_experience > 0:
            #     strategy.loss += self.trip_loss(strategy)
            old_representation = self.old_model(strategy.mb_x)[0]
            new_representation = strategy.mb_output
            alpha = 1.0
            beta = 8.0
            temperature = 4.0
            warmup = 20
            dkd = min(strategy.clock.train_exp_epochs / warmup, 1.0)
            # old_representation = torch.cat(tuple(old_representation), dim=0)
            # new_representation = torch.cat(tuple(new_representation), dim=0)
            strategy.loss += dkd * self.dkd_loss(new_representation, old_representation, strategy.mb_y, alpha, beta,
                                                 temperature)
    def dkd_loss(self, logits_student, logits_teacher, target, alpha, beta, temperature):
        gt_mask = self._get_gt_mask(logits_student, target)
        other_mask = self._get_other_mask(logits_student, target)
        pred_student = F.softmax(logits_student / temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
        pred_student = self.cat_mask(pred_student, gt_mask, other_mask)
        pred_teacher = self.cat_mask(pred_teacher, gt_mask, other_mask)
        log_pred_student = torch.log(pred_student)
        # print('loss initialize ok')
        tckd_loss = (
                F.kl_div(log_pred_student, pred_teacher)
                * (temperature ** 2)
                / target.shape[0]
        )
        # print("tckd_loss is ok")
        pred_teacher_part2 = F.softmax(
            logits_teacher / temperature - 1000.0 * gt_mask, dim=1
        )
        log_pred_student_part2 = F.log_softmax(
            logits_student / temperature - 1000.0 * gt_mask, dim=1
        )
        nckd_loss = (
                F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
                * (temperature ** 2)
                / target.shape[0]
        )
        # print("nckd_loss is ok")
        return alpha * tckd_loss + beta * nckd_loss

    def _get_gt_mask(self, logits, target):
        # print(target.size())
        target = target.reshape(-1)
        # print(target.size())
        mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
        # print(mask.size())
        return mask

    def _get_other_mask(self, logits, target):
        target = target.reshape(-1)
        mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
        return mask

    def cat_mask(self, t, mask1, mask2):
        t1 = (t * mask1).sum(dim=1, keepdims=True)
        t2 = (t * mask2).sum(1, keepdims=True)
        rt = torch.cat([t1, t2], dim=1)
        return rt


