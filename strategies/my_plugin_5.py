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


class MyPlugin5(SupervisedPlugin):
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
        self.eval_output = []
        self.exp_y = None
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.batch_size_mem = batch_size_mem
        self.redius = 0
        self.protos = []
        self.acc_epoch_list = [[] for i in range(50)]
        self.loss_epoch_list = [[] for j in range(50)]
        self.proto_labels = []
        self.count_e = defaultdict(int)
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
        strategy.mbatch[0] = torch.stack([strategy.mbatch[0], transforms.RandomHorizontalFlip(p=1)(strategy.mbatch[0])], 1)
        strategy.mbatch[0] = strategy.mbatch[0].view(-1, 3, 32,32)
        strategy.mbatch[1] = torch.stack([strategy.mbatch[1] for k in range(2)], 1).view(-1)


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
        """清空之前exp保留的数据"""
        self.exp_x = []
        self.exp_y = []
        self.eval_exp_y = []
        self.eval_output = []
        self.eval_output = []
        # strategy.model.linear.sigma.requires_grad = False
        # strategy.model.linear.bias.requires_grad = False
        """加载bestModel"""
        # if strategy.experience.current_experience > 0:
        #     strategy.model = self.un_freeze_model(self.bestModel)

        """根据每个exp新旧class的数目动态改变每个exp需要训练的epoch数"""
        if strategy.experience.current_experience == 0:
            self.dic = defaultdict(int)
            for i in strategy.benchmark.first_occurrences:
                self.dic[i] += 1
        new_class_in_curr_exp = self.dic[strategy.experience.current_experience]
        old_class_in_curr_exp = strategy.benchmark.present_classes_in_each_exp[strategy.experience.current_experience].shape[0] - new_class_in_curr_exp

        if strategy.experience.current_experience == 0:
            if new_class_in_curr_exp >= 30:
                self.flag = True

        if strategy.experience.current_experience == 10:
            strategy.train_epochs = 50
        if strategy.experience.current_experience == 20:
            strategy.train_epochs = 40
        if strategy.experience.current_experience == 30:
            strategy.train_epochs = 30
        if strategy.experience.current_experience == 40:
            strategy.train_epochs = 20


    def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        # if strategy.experience.current_experience >= 48:
        #     print(strategy.model.linear.sigma)
        # print(strategy.model.w1)
        if strategy.experience.current_experience == 49:
            print(self.count_e)
            print((strategy.model.linear.weight[:,:]**2).sum(1).sqrt().detach().cpu().numpy())
        representation = []
        for batch_data in self.exp_x:
            with torch.no_grad():
                batch_representation = strategy.model(batch_data)[1]
                batch_representation = torch.nn.functional.normalize(batch_representation)
                representation.append(batch_representation)
        representation = torch.cat(tuple(representation), dim=0)
        self.exp_y = torch.cat(tuple(self.exp_y), dim=0)
        # strategy.model.linear.sigma.requires_grad = True
        # strategy.model.linear.bias.requires_grad = True

        """
        如果当前experience>0,则进行漂移补偿，并使用补偿后的protos训练网络
        使用old_model和model提取当前任务数据的表征
        """
        length = strategy.experience.current_experience // 10 + 1 if self.flag else 2
        if strategy.experience.current_experience > 0:
            # old_representation = []
            # for batch_data in self.exp_x:
            #     with torch.no_grad():
            #         batch_representation = self.old_model(batch_data)[1]
            #         batch_representation = torch.nn.functional.normalize(batch_representation)
            #         old_representation.append(batch_representation)
            # old_representation = torch.cat(tuple(old_representation), dim=0)
            #
            # MU = np.asarray(deepcopy(self.protos))
            # # print("补偿之前的原型：", MU)
            # gap = self.displacement(old_representation, representation, MU, 0.2)
            # MU += gap
            # print("补偿之后的原型：", MU)
            # self.protos = MU.tolist()

            """protos训练"""
            for i in range(length):
                loss_proto_before_save_cur_protos = self.compute_loss(strategy).requires_grad_(True)  # * 0.3
                strategy.optimizer.zero_grad()
                loss_proto_before_save_cur_protos.backward()
                strategy.optimizer.step()


        self.protoSave(representation)

        """protos训练"""
        for i in range(length):
            loss_proto_after_save_cur_protos = self.compute_loss(strategy).requires_grad_(True)  # * 0.3
            strategy.optimizer.zero_grad()
            loss_proto_after_save_cur_protos.backward()
            strategy.optimizer.step()
        # strategy.model.linear.sigma.requires_grad = False
        # strategy.model.linear.bias.requires_grad = False
        """保存模型并冻结"""
        self.old_model = deepcopy(strategy.model)
        self.old_model = self.freeze_model(self.old_model)
        if strategy.experience.current_experience == 49:
            print("分类器权重")
            print((strategy.model.linear.weight[:, :] ** 2).sum(1).sqrt().detach().cpu().numpy())
            print("缩放权重")
            # print(strategy.model.linear.sigma)

    def displacement(self, Y1, Y2, embedding_old, sigma):
        Y1 = Y1.cpu()
        Y2 = Y2.cpu()
        DY = Y2 - Y1  # 5000 x 512
        distance = np.sum((np.tile(Y1[None, :, :], [embedding_old.shape[0], 1, 1]) -
                           np.tile(embedding_old[:, None, :], [1, Y1.shape[0], 1])) ** 2,
                          axis=2)  # 50 x 5000； 计算embedding_old中每个样本与Y1之间的距离
        W = np.exp(-distance / (2 * sigma ** 2))  # 50 x 5000; # +1e-5
        W_norm = W / np.tile(np.sum(W, axis=1)[:, None], [1, W.shape[1]])  # 50 x 5000; #  对 W 进行归一化以得到 W_norm
        # 将 W_norm 和 DY 相乘，获得每个样本的位移，并将这些位移求和以获得总位移。
        displacement = np.sum(
            np.tile(W_norm[:, :, None], [1, 1, DY.shape[1]]) * np.tile(DY[None, :, :], [W.shape[0], 1, 1]), axis=1)
        return displacement  # 50 x 512

    def protoSave(self, feature):  # feature (2000,512)
        features = []
        labels = []
        for i in range(feature.shape[0]):
            features.append(feature[i].cpu().numpy())
            labels.append(int(self.exp_y[i]))

        labels_set = np.unique(labels)  #  !! 消除重复的标签，得到本次所要存储的所有类标签
        features = np.array(features)  # ！！将列表转为数组
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


    def after_training_iteration(self, strategy, **kwargs):
        """
        Example callback: before backward.
        """
        benchmark = strategy.benchmark
        for i in strategy.mb_y.cpu().numpy().tolist():
            self.count_e[i] += 1
        if strategy.clock.train_exp_epochs == 0:
            self.exp_x.append(strategy.mb_x)
            self.exp_y.append(strategy.mb_y)
            self.new_S = self.compute_buffer(strategy, benchmark.num_seen_class)
        pass


    def compute_buffer(self, stragey, num_seen_class):
        '''
        num_seen_class : 当前已知类的每类的样本数量
        '''
        for c in stragey.mb_y:
            if c.item() in num_seen_class:
                num_seen_class[c.item()] += 1
            else:
                num_seen_class[c.item()] = 1
        # print(num_seen_class)
        q = {}
        for i in num_seen_class:
            q[i] = num_seen_class[i]
        m = sum(q.values())
        S = {}
        for c in q:
            S[c] = q[c] / m
        return S

    # @torch.no_grad()
    def compute_loss(self, strategy):
        # print(self.new_S)
        loss_protoaug = torch.tensor(0.0).to(strategy.device)
        for i in range(len(self.protos)):
            proto = torch.tensor(self.protos[i]).to(strategy.device)
            alpha = 100 * torch.tensor(self.new_S[self.proto_labels[i]]).to(strategy.device) # 10
            # alpha = 1
            soft_feat = strategy.model.linear(proto.reshape(1,-1)).to(strategy.device).reshape(-1)
            loss_protoaug += alpha * nn.CrossEntropyLoss()(soft_feat, torch.tensor(self.proto_labels[i]).to(strategy.device))   #alpha *

        return loss_protoaug

    def after_eval_forward(
        self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:
            pass



    def after_eval_iteration(
            self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:
        self.eval_exp_y.append(strategy.mb_y)
        self.eval_output.append(strategy.mb_output)

    def after_eval_exp(
            self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:
        correct = defaultdict(int)
        total = defaultdict(int)
        acc_class_list = defaultdict(int)
        pre = torch.cat(tuple([torch.max(i, 1)[1] for i in self.eval_output])).cpu().numpy().tolist()
        self.eval_exp_y = torch.cat(tuple(self.eval_exp_y)).cpu().numpy().tolist()
        for i in range(strategy.benchmark.n_classes):
            correct[i] = 0
        for pre, label in zip(pre, self.eval_exp_y):
            if pre == label:
                correct[pre]+=1
            total[label]+=1

        for i in correct:
            acc_class_list[i] = correct[i]/total[i]
        print("每个类的准确率为")
        for i in acc_class_list:
            print(i,acc_class_list[i])


        """将loss和acc写入列表"""
        acc_experience = strategy.evaluator.metrics[2]._metric._mean_accuracy.summed / strategy.evaluator.metrics[
            2]._metric._mean_accuracy.weight
        loss_experience = strategy.evaluator.metrics[6]._metric._mean_loss.summed / strategy.evaluator.metrics[
            6]._metric._mean_loss.weight
        self.acc_experience_list.append(acc_experience)
        self.loss_experience_list.append(loss_experience)

        """将loss和acc列表保存至文件"""
        if strategy.experience.current_experience == 49:
            title = "nonet.txt"
            """将训练好的model保存至文件"""
            with open('nonet.pkl', 'wb') as f:
                pickle.dump(strategy.model, f)
            text_txt = 'acc1'+title
            with open(text_txt, 'a') as text:
                s = ''
                for i in range(len(self.acc_experience_list)):
                    s += str(self.acc_experience_list[i]) + ' ' if i != len(self.acc_experience_list) - 1 else str(
                        self.acc_experience_list[i]) + '\n'
                text.write(s)
            text_txt = 'acc'+title
            with open(text_txt, 'a') as text:
                s = ''
                for i in acc_class_list:
                    s += str(acc_class_list[i])
                text.write(s)
            text_txt = 'loss'+title
            with open(text_txt, 'a') as text:
                s = ''
                for i in range(len(self.loss_experience_list)):
                    s += str(self.loss_experience_list[i]) + ' ' if i != len(self.loss_experience_list) - 1 else str(
                        self.loss_experience_list[i]) + '\n'
                text.write(s)
        self.count += 1
        # s = strategy.model.linear.weight
        # alpha = torch.tensor(self.new_S[self.proto_labels[i]]).to(strategy.device)
