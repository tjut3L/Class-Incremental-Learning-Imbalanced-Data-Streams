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
from torch.utils.data import DataLoader, Dataset
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


class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        assert len(features) == len(labels), "Data size error!"
        self.features = torch.from_numpy(features)
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]

        return idx, feature, label

class FetrlPlugin(SupervisedPlugin):
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
        self._means = {}
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

        self.known = []
        self.unknown = []

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

    def compute_means(self, feature):
        features = []
        labels = []
        self.unknown = []
        for i in range(feature.shape[0]):
            features.append(feature[i].cpu().numpy())
            labels.append(int(self.exp_y[i]))
        labels_set = np.unique(labels)  # 消除重复的标签，得到本次所要存储的所有类标签
        features = np.array(features)  # 将列表转为数组
        # difference_labels = set(self.proto_labels) - set(labels_set)
        unknown = []
        for i in range(100):
            if i not in self.known:
                unknown.append(i)
        for label in labels_set:
            if label in unknown:
                index = np.where(label == labels)[0]
                # 获得某一类别在所有标签列表的索引位置，再根据这个索引位置再所有的特征列表中找到某一类别的所有图像特征,
                # np.where（）返回一个元组，元组里面是个数组，这个就是索引位置
                class_feature = features[index]
                class_mean = np.mean(class_feature, axis=0)
                self._means[label] = class_mean
                self.proto_labels.append(label)
                self.unknown.append(label)

    def _compute_relations(self):
        old_means = np.array([self._means[i] for i in self.known])
        new_means = np.array([self._means[i] for i in self.unknown])
        self._relations = np.argmax((old_means / np.linalg.norm(old_means, axis=1)[:, None]) @ (
                    new_means / np.linalg.norm(new_means, axis=1)[:, None]).T, axis=1) + len(self.proto_labels)

    def _build_feature_set(self, strategy):
        self.vectors_train = []
        self.labels_train = []
        _known_classes = len(self.known)
        _total_classes = 100
        for class_idx in self.known:
            new_idx = self._relations[class_idx]
            self.vectors_train.append(
                self.vectors_train[new_idx - _known_classes] - self._means[new_idx] + self._means[class_idx])
            self.labels_train.append([class_idx] * len(self.vectors_train[-1]))
        self.vectors_train = np.concatenate(self.vectors_train)
        self.labels_train = np.concatenate(self.labels_train)



    def before_training_exp(
            self,
            strategy: "SupervisedTemplate",
            num_workers: int = 0,
            shuffle: bool = True,
            **kwargs
    ):

        """清空之前exp保留的数据"""
        self.exp_x = []
        self.exp_y = []

    def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        if strategy.experience.current_experience > 0:
            strategy.optimizer = torch.optim.Adam(strategy.model.linear.parameters(), lr=0.001)
        representation = []
        for batch_data in self.exp_x:
            with torch.no_grad():
                batch_representation = strategy.model(batch_data)[1]
                batch_representation = torch.nn.functional.normalize(batch_representation)
                representation.append(batch_representation)
        representation = torch.cat(tuple(representation), dim=0)

        self.exp_y = torch.cat(tuple(self.exp_y), dim=0)
        self.old_model = deepcopy(strategy.model)
        self.old_model = self.freeze_model(self.old_model)
        self.compute_means(representation)
        if strategy.experience.current_experience > 0:
            self._compute_relations()
            self._build_feature_set(strategy)
            loss = 0.0
            for i in range(len(self.vectors_train)):
                pre = strategy.model.linear(i)
                c_loss = F.cross_entropy(pre,self.labels_train[i])
                # loss+=c_loss



        for i in self.unknown:
            self.known.append(i)




    def after_training_iteration(self, strategy, **kwargs):
        """
        Example callback: before backward.
        """
        if strategy.clock.train_exp_epochs == 0:
            self.exp_x.append(strategy.mb_x)
            self.exp_y.append(strategy.mb_y)
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
