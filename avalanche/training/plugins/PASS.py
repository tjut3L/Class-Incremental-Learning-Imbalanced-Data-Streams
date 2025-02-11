import pickle
from collections import defaultdict
from typing import Optional, TYPE_CHECKING, List
import copy
import os

from sklearn.cluster import KMeans
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


class PASSPlugin(SupervisedPlugin):
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
        self.radius = 0
        self.exp_y = None
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.batch_size_mem = batch_size_mem
        self.redius = 0
        self.protos = []
        self.acc_epoch_list = [[] for i in range(50)]
        self.loss_epoch_list = [[] for j in range(50)]
        self.proto_labels = []
        self.old_model = None
        self.new_S = {}
        self.eval_count = 0
        self.old_model1 = None
        self.flag = False
        self.bestModel = None
        self.clock_class = None
        self.old_bias = {}
        self.old_weight = {}
        self.acc_experience_list = []
        self.loss_experience_list = []
        self.eval_exp_x = []
        self.eval_exp_y = []
        self.count = 0
        if storage_policy is not None:  # Use other storage policy
            self.storage_policy = storage_policy
            assert storage_policy.max_size == self.mem_size
        else:  # Default
            self.storage_policy = ERBuffer(
                max_size=self.mem_size,
            )

    def before_training_iteration(
            self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:
        # pass
        # strategy.mbatch[0] = torch.stack([strategy.mbatch[0], transforms.RandomHorizontalFlip(p=1)(strategy.mbatch[0]),transforms.RandomVerticalFlip(p=1)(strategy.mbatch[0])], 1)
        strategy.mbatch[0] = torch.stack([torch.rot90(strategy.mbatch[0], k, (2, 3)) for k in range(4)], 1)
        strategy.mbatch[0] = strategy.mbatch[0].view(-1, 3, 32, 32)
        strategy.mbatch[1] = torch.stack([strategy.mbatch[1] * 4 + k for k in range(4)], 1).view(-1)

    def freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False
        return model

    def un_freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = True
        return model

    def protoSave(self,strategy, feature):  # feature (2000,512)
        features = []
        labels = []
        for i in range(feature.shape[0]):
            features.append(feature[i].cpu().numpy())
            labels.append(int(self.exp_y[i]))
        radius = []
        feature_dim = feature.shape[1]
        labels_set = np.unique(labels)  # 消除重复的标签，得到本次所要存储的所有类标签
        features = np.array(features)  # 将列表转为数组
        # difference_labels = set(self.proto_labels) - set(labels_set)
        for label in labels_set:
            index = np.where(label == labels)[0]  # 获得某一类别在所有标签列表的索引位置，再根据这个索引位置再所有的特征列表中找到某一类别的所有图像特征,
            # np.where（）返回一个元组，元组里面是个数组，这个就是索引位置
            class_feature = features[index]
            if label not in self.proto_labels:
                self.protos.append(np.mean(class_feature, axis=0))  # 计算这一类的所有特征的均值并将其存储在proto列表里
                self.proto_labels.append(label)
            else:
                idx = self.proto_labels.index(label)
                self.protos[idx] = np.mean(class_feature, axis=0)
            if strategy.experience.current_experience == 0:
                cov = np.cov(class_feature.T)
                radius.append(np.trace(cov) / feature_dim)
                self.radius = np.sqrt(np.mean(radius))


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
        """清空之前exp保留的数据"""
        self.exp_x = []
        self.exp_y = []

    def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        representation = []
        for batch_data in self.exp_x:
            with torch.no_grad():
                batch_representation = strategy.model(batch_data)[1]
                batch_representation = torch.nn.functional.normalize(batch_representation)
                representation.append(batch_representation)
        representation = torch.cat(tuple(representation), dim=0)
        self.exp_y = torch.cat(tuple(self.exp_y), dim=0)
        self.protoSave(strategy, representation)
        self.old_model = deepcopy(strategy.model)
        self.old_model = self.freeze_model(self.old_model)

    def _compute_loss(self, strategy):
        temp1 = 0.1
        protoAug_weight = 10.0
        kd_weight = 10.0
        old_class = len(self.protos)
        imgs = strategy.mb_x
        target = strategy.mb_y
        output = strategy.model(imgs)[0]
        output, target = output.to(strategy.device), target.to(strategy.device)
        loss_cls = nn.CrossEntropyLoss()(output / temp1, target)
        if self.old_model is None:
            return loss_cls
        else:
            feature = strategy.model(imgs)[1]
            feature_old = self.old_model(imgs)[1]
            loss_kd = torch.dist(feature, feature_old, 2)

            proto_aug = []
            proto_aug_label = []
            index = list(range(old_class))
            for _ in range(strategy.train_mb_size*4):
                np.random.shuffle(index)
                temp = self.protos[index[0]] + np.random.normal(0, 1, 512) * self.radius
                proto_aug.append(temp)
                proto_aug_label.append(self.proto_labels[index[0]])

            proto_aug = torch.from_numpy(np.float32(np.asarray(proto_aug))).float().to(strategy.device)
            proto_aug_label = torch.from_numpy(np.asarray(proto_aug_label)).to(strategy.device)
            soft_feat_aug = strategy.model.linear(proto_aug)
            loss_protoAug = nn.CrossEntropyLoss()(soft_feat_aug / temp1, proto_aug_label)

            return loss_cls + protoAug_weight * loss_protoAug + kd_weight * loss_kd

    def after_training_iteration(self, strategy, **kwargs):
        """
        Example callback: before backward.
        """
        if strategy.clock.train_exp_epochs == 0:
            self.exp_x.append(strategy.mb_x)
            self.exp_y.append(strategy.mb_y)

    def before_backward(
            self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:
        strategy.loss += self._compute_loss(strategy)
        pass
    def before_eval_iteration(
        self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:
        for i in range(len(strategy.mb_y)):
            strategy.mb_y[i] *=4
        pass
    def after_eval_iteration(
            self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:
        if self.count == 0:
            self.eval_exp_x.append(strategy.mb_x)
            self.eval_exp_y.append(strategy.mb_y)

    def after_eval_exp(
            self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:

        """将loss和acc写入列表"""
        acc_experience = strategy.evaluator.metrics[2]._metric._mean_accuracy.summed / strategy.evaluator.metrics[
            2]._metric._mean_accuracy.weight
        loss_experience = strategy.evaluator.metrics[6]._metric._mean_loss.summed / strategy.evaluator.metrics[
            6]._metric._mean_loss.weight
        self.acc_experience_list.append(acc_experience)
        self.loss_experience_list.append(loss_experience)

        """将loss和acc列表保存至文件"""
        if self.count == 49:
            text_txt = 'acc_s3_PASS.txt'
            with open(text_txt, 'a') as text:
                s = ''
                for i in range(len(self.acc_experience_list)):
                    s += str(self.acc_experience_list[i]) + ' ' if i != len(self.acc_experience_list) - 1 else str(
                        self.acc_experience_list[i]) + '\n'
                text.write(s)
            text_txt = 'loss_s3_PASS.txt'
            with open(text_txt, 'a') as text:
                s = ''
                for i in range(len(self.loss_experience_list)):
                    s += str(self.loss_experience_list[i]) + ' ' if i != len(self.loss_experience_list) - 1 else str(
                        self.loss_experience_list[i]) + '\n'
                text.write(s)
            """将训练好的model保存至文件"""
            # with open('Our_cub_2_all.pkl', 'wb') as f:
            #     pickle.dump(strategy.model, f)
        self.count += 1
