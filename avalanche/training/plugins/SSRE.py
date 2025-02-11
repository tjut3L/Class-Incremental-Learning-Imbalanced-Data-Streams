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
from utils.model import backbones
if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate


class SSREPlugin(SupervisedPlugin):
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
        self.cur_exp = 0
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
        self.prototype = []
        self.class_label = []
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
        labels_set = np.unique(labels)
        features = np.array(features)
        for label in labels_set:
            index = np.where(label == labels)[0]
            class_feature = features[index]
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
        self.cur_exp = strategy.experience.current_experience
        # if strategy.experience.current_experience == 0:
        #     strategy.model.eval()
        #     para = torch.load('/home/tjut_zhaoyishuo/clvision-challenge-23/utils/model/model_imagenet_10.pth')
        #     para_dict = para['state_dict']
        #     para_dict_re = self.structure_reorganization(para_dict)
        #     model_dict = strategy.model.state_dict()
        #     state_dict = {k: v for k, v in para_dict_re.items() if k in model_dict.keys()}
        #     model_dict.update(state_dict)
        #     strategy.model.load_state_dict(model_dict)
        if strategy.experience.current_experience > 0:
                strategy.model.eval()
                backbone = backbones['resnet18_no1'](mode='parallel_adapters').to(strategy.device)
                model_dict = backbone.state_dict()
                para_dict = strategy.model.backbone.state_dict()
                state_dict = {k: v for k, v in para_dict.items() if k in model_dict.keys()}
                model_dict.update(state_dict)
                backbone.load_state_dict(model_dict)
                strategy.model.backbone = backbone
                strategy.model.fix_backbone_adapter()


    def structure_reorganization(self, para_dict):
        para_dict_re = copy.deepcopy(para_dict)
        for k, v in para_dict.items():
            if 'bn.weight' in k or 'bn1.weight' in k or 'downsample.1.weight' in k:
                if 'bn.weight' in k:
                    k_conv3 = k.replace('bn', 'conv')
                elif 'bn1.weight' in k:
                    k_conv3 = k.replace('bn1', 'conv1')
                elif 'downsample.1.weight' in k:
                    k_conv3 = k.replace('1', '0')
                k_conv3_bias = k_conv3.replace('weight', 'bias')
                k_bn_bias = k.replace('weight', 'bias')
                k_bn_mean = k.replace('weight', 'running_mean')
                k_bn_var = k.replace('weight', 'running_var')

                gamma = para_dict[k]
                beta = para_dict[k_bn_bias]
                running_mean = para_dict[k_bn_mean]
                running_var = para_dict[k_bn_var]
                eps = 1e-5
                std = (running_var + eps).sqrt()
                t = (gamma / std).reshape(-1, 1, 1, 1)
                para_dict_re[k_conv3] *= t
                para_dict_re[k_conv3_bias] = beta - running_mean * gamma / std
        return para_dict_re


    def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        representation = []
        for batch_data in self.exp_x:
            with torch.no_grad():
                batch_representation = strategy.model.feature_extractor(batch_data)
                batch_representation = torch.nn.functional.normalize(batch_representation)
                representation.append(batch_representation)
        representation = torch.cat(tuple(representation), dim=0)
        self.exp_y = torch.cat(tuple(self.exp_y), dim=0)
        self.protoSave(strategy,representation)
        self.old_model = deepcopy(strategy.model)
        self.old_model = self.freeze_model(self.old_model)
        self.old_model = strategy.model.copy()
        self.old_model.to(strategy.device)
        self.old_model.eval()

        if strategy.experience.current_experience > 0:
            model_dict = strategy.model.state_dict()
            for k, v in model_dict.items():
                if 'adapter' in k:
                    k_conv3 = k.replace('adapter', 'conv')
                    model_dict[k_conv3] = model_dict[k_conv3] + F.pad(v, [1, 1, 1, 1], 'constant', 0)
                    model_dict[k] = torch.zeros_like(v)
            strategy.model.load_state_dict(model_dict)

    def _compute_loss(self, strategy):
        old_class = len(self.protos)
        imgs = strategy.mb_x
        target = strategy.mb_y
        if self.old_model is None:
            output = strategy.model(imgs)
            loss_cls = nn.CrossEntropyLoss()(output / 0.1, target)
            return loss_cls
        else:
            feature = strategy.model.feature_extractor(imgs)
            feature_old = self.old_model.feature_extractor(imgs)

            proto = torch.from_numpy(np.array(self.protos)).t().to(strategy.device)
            proto_nor = torch.nn.functional.normalize(proto, p=2, dim=0, eps=1e-12)
            feature_nor = torch.nn.functional.normalize(feature, p=2, dim=-1, eps=1e-12)
            cos_dist = feature_nor @ proto_nor
            cos_dist = torch.max(cos_dist, dim=-1).values
            cos_dist2 = 1 - cos_dist
            output = strategy.model(imgs)
            loss_cls = nn.CrossEntropyLoss(reduce=False)(output / 0.1, target)
            loss_cls = torch.mean(loss_cls * cos_dist2, dim=0)

            loss_kd = torch.norm(feature - feature_old, p=2, dim=1)
            loss_kd = torch.sum(loss_kd * cos_dist, dim=0)

            proto_aug = []
            proto_aug_label = []
            index = list(range(old_class))
            for _ in range(strategy.train_mb_size * 4):
                np.random.shuffle(index)
                temp = self.protos[index[0]] + np.random.normal(0, 1, 512) * self.radius
                proto_aug.append(temp)
                proto_aug_label.append(self.proto_labels[index[0]])

            proto_aug = torch.from_numpy(np.float32(np.asarray(proto_aug))).float().to(strategy.device)
            proto_aug_label = torch.from_numpy(np.asarray(proto_aug_label)).to(strategy.device)
            soft_feat_aug = strategy.model.classify(proto_aug)
            loss_protoAug = nn.CrossEntropyLoss()(soft_feat_aug / 0.1, proto_aug_label)
            return loss_cls + 10 * loss_protoAug + loss_kd

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
    def after_eval_iteration(
            self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:
        if self.count == 0:
            self.eval_exp_x.append(strategy.mb_x)
            self.eval_exp_y.append(strategy.mb_y)
    def after_training_epoch(
        self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:
        strategy.scheduler.step()
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
            text_txt = 'acc_s1_SSRE.txt'
            with open(text_txt, 'a') as text:
                s = ''
                for i in range(len(self.acc_experience_list)):
                    s += str(self.acc_experience_list[i]) + ' ' if i != len(self.acc_experience_list) - 1 else str(
                        self.acc_experience_list[i]) + '\n'
                text.write(s)
            text_txt = 'loss_s1_SSRE.txt'
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
