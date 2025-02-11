import pickle
from collections import defaultdict
from typing import Optional, TYPE_CHECKING, List
import copy
import os
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
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


class PRAKAPlugin(SupervisedPlugin):
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
        self.scheduler = None
        self.kd_weight = 1.5
        self.protoAug_weight = 1.5
        self.temp = 0.1
        self.joint_labels = None
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
        self.single_label = None
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
        strategy.mbatch[0] = strategy.mbatch[0].view(-1, 3, strategy.mbatch[0].size()[-1], strategy.mbatch[0].size()[-1])
        self.joint_labels = torch.stack([strategy.mbatch[1] * 4 + k for k in range(4)], 1).view(-1)
        self.single_label = deepcopy(strategy.mbatch[1])
        strategy.mbatch[1] = torch.stack([strategy.mbatch[1] for k in range(4)], 1).view(-1)



    def freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False
        return model

    def un_freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = True
        return model

    def protoSave(self, strategy, feature):  # feature (2000,512)
        features = []
        labels = []
        for i in range(feature.shape[0]):
            features.append(feature[i].cpu().numpy())
            labels.append(int(self.exp_y[i]))

        feature_dim = feature.shape[1]
        labels_set = np.unique(labels)  # 消除重复的标签，得到本次所要存储的所有类标签
        features = np.array(features)  # 将列表转为数组
        # difference_labels = set(self.proto_labels) - set(labels_set)
        for label in labels_set:
            index = np.where(label == labels)[0]  # 获得某一类别在所有标签列表的索引位置，再根据这个索引位置再所有的特征列表中找到某一类别的所有图像特征,
            # np.where（）返回一个元组，元组里面是个数组，这个就是索引位置
            i = 0
            index1 = []
            while i < len(index):
                index1.append(index[i])
                i += 4
            class_feature = features[index1]
            if label not in self.proto_labels:
                self.protos.append(np.mean(class_feature, axis=0))  # 计算这一类的所有特征的均值并将其存储在proto列表里
                self.proto_labels.append(label)
            else:
                idx = self.proto_labels.index(label)
                self.protos[idx] = np.mean(class_feature, axis=0)

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

        # if strategy.experience.current_experience == 10:
        #     strategy.train_epochs = 50
        # if strategy.experience.current_experience == 20:
        #     strategy.train_epochs = 40
        # if strategy.experience.current_experience == 30:
        #     strategy.train_epochs = 30
        # if strategy.experience.current_experience == 40:
        #     strategy.train_epochs = 20
        self.scheduler = CosineAnnealingLR(strategy.optimizer, T_max=7500)

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
        self.freeze_model(self.old_model)
    def _compute_loss(self, strategy):
        feature = strategy.model(strategy.mb_x)[1]
        joint_preds = strategy.model.fc(feature)
        single_preds = strategy.model.linear(feature)[::4]
        joint_preds, joint_labels, single_preds, labels = joint_preds.to(strategy.device), self.joint_labels.to(
            strategy.device), single_preds.to(strategy.device), self.single_label
        joint_loss = nn.CrossEntropyLoss()(joint_preds / self.temp, joint_labels)
        signle_loss = nn.CrossEntropyLoss()(single_preds / self.temp, labels)

        agg_preds = 0
        for i in range(4):
            agg_preds = agg_preds + joint_preds[i::4, i::4] / 4

        distillation_loss = F.kl_div(F.log_softmax(single_preds, 1),
                                     F.softmax(agg_preds.detach(), 1),
                                     reduction='batchmean')

        if strategy.experience.current_experience == 0:
            return joint_loss + signle_loss + distillation_loss
        else:
            # feature_old = self.old_model(strategy.mb_x)[1]
            # loss_kd = torch.dist(feature, feature_old, 2)

            proto_aug = []
            proto_aug_label = []
            old_class_list = self.proto_labels
            for _ in range(feature.shape[0] // 4):  # batch_size = feature.shape[0] // 4
                i = np.random.randint(0, feature.shape[0])
                np.random.shuffle(old_class_list)
                lam = np.random.beta(0.5, 0.5)
                if lam > 0.6:
                    # lam = 1 - lam
                    lam = lam * 0.6
                if np.random.random() >= 0.5:
                    temp = (1 + lam) * self.protos[self.proto_labels.index(old_class_list[0])] - lam * feature.detach().cpu().numpy()[i]
                else:
                    temp = (1 - lam) * self.protos[self.proto_labels.index(old_class_list[0])] + lam * feature.detach().cpu().numpy()[i]

                proto_aug.append(temp)
                proto_aug_label.append(old_class_list[0])

            proto_aug = torch.from_numpy(np.float32(np.asarray(proto_aug))).float().to(strategy.device)
            proto_aug_label = torch.from_numpy(np.asarray(proto_aug_label)).to(strategy.device)
            aug_preds = strategy.model.linear(proto_aug)
            joint_aug_preds = strategy.model.fc(proto_aug)

            agg_preds = 0
            agg_preds = agg_preds + joint_aug_preds[:, ::4]

            aug_distillation_loss = F.kl_div(F.log_softmax(aug_preds, 1),
                                             F.softmax(agg_preds.detach(), 1),
                                             reduction='batchmean')
            loss_protoAug = nn.CrossEntropyLoss()(aug_preds / self.temp, proto_aug_label) + nn.CrossEntropyLoss()(
                joint_aug_preds / self.temp, proto_aug_label * 4) + aug_distillation_loss

            return joint_loss + signle_loss + distillation_loss + self.protoAug_weight * loss_protoAug

    def after_training_iteration(self, strategy, **kwargs):
        """
        Example callback: before backward.
        """
        if strategy.clock.train_exp_epochs == 0:
            self.exp_x.append(strategy.mb_x)
            self.exp_y.append(strategy.mb_y)
        self.scheduler.step()

    def before_backward(
            self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:
        strategy.loss += self._compute_loss(strategy)
        pass

    # def before_eval_iteration(
    #     self, strategy: Template, *args, **kwargs
    # ) -> CallbackResult:
    #     for i in range(len(strategy.mb_y)):
    #         strategy.mb_y[i] *=4
    #     pass
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
            text_txt = 'acc_s3_PRAKA.txt'
            with open(text_txt, 'a') as text:
                s = ''
                for i in range(len(self.acc_experience_list)):
                    s += str(self.acc_experience_list[i]) + ' ' if i != len(self.acc_experience_list) - 1 else str(
                        self.acc_experience_list[i]) + '\n'
                text.write(s)
            text_txt = 'loss_s3_PRAKA.txt'
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
