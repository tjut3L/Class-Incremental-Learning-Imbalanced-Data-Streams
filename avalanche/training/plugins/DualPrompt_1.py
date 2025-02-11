import pickle
import random
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

import xlsxwriter as xw

if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate


class DualPlugin(SupervisedPlugin):
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
        self.eval_exp_x = []
        self.eval_exp_y = []
        self.eval_output = []
        self.exp_y = None
        self.proto_labels = []
        self.protos = []
        self.acc_l = []
        self.exp_y = []
        self.exp_x = []
        self.count = 0
        self.new_S = None
        self.cur_exp = 0
        self.n_classes = 100
        self.old_prompt = None
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
        super().__init__()

    def before_training_iteration(

            self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:
        if self.new_S is not None:
            strategy.model.e_prompt.Q = self.new_S
        with torch.no_grad():
            if strategy.original_model is not None:
                output = strategy.original_model(strategy.mb_x)
                strategy.cls_features = output['pre_logits']
            else:
                strategy.cls_features = None
        pass

    # def after_backward(
    #     self, strategy: Template, *args, **kwargs
    # ) -> CallbackResult:
    #     prompt_loss = torch.dist(self.rompt.prompt, p=2)
    #     #     print("prompt_loss: ",prompt_loss)old_prompt, strategy.model.e_p
    #     strategy.optimizer.zero_grad()
    #     prompt_loss.backward()
    #     strategy.optimizer.step()
    def before_backward(
            self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:
        class_mask = strategy.benchmark.present_classes_in_each_exp
        if strategy.args.train_mask and class_mask is not None:
            mask = class_mask[self.cur_exp]
            not_mask = np.setdiff1d(np.arange(self.n_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(strategy.device)
            strategy.mb_output = strategy.mb_output.index_fill(dim=1, index=not_mask, value=float('-inf'))
        if strategy.args.pull_constraint and 'reduce_sim' in strategy.output_list:
            strategy.loss = strategy.loss - strategy.args.pull_constraint_coeff * strategy.output_list['reduce_sim']

    def before_training_epoch(
            self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:
        strategy.model.train(strategy.args.set_training_mode)
        strategy.original_model.eval()

        pass

    def protoSave(self, feature):  # feature (2000,768)
        features = []
        labels = []
        for i in range(feature.shape[0]):
            features.append(feature[i].cpu().numpy())
            labels.append(int(self.exp_y[i]))
        feature_dim = feature.shape[1]
        labels_set = np.unique(labels)  # 消除重复的标签，得到本次所要存储的所有类标签
        features = np.array(features)  # 将列表转为数组
        # difference_labels = set(self.proto_labels) - set(labels_set)

        class_label = []

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
        return

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
        # self.cur_exp = strategy.experience.current_experience
        self.exp_y = []
        self.exp_x = []
        self.eval_exp_y = []
        self.eval_exp_x = []
        self.eval_output = []
        self.eval_output = []
        # if strategy.experience.current_experience >= 15:
        #     strategy.plugins[0].lwf.alpha = 30.0
        self.old_prompt = copy.deepcopy(strategy.model.e_prompt.prompt.detach())
        if strategy.args.prompt_pool and strategy.args.shared_prompt_pool:
            if self.cur_exp > 0:
                prev_start = (self.cur_exp - 1) * strategy.args.top_k
                prev_end = self.cur_exp * strategy.args.top_k

                cur_start = prev_end
                cur_end = (self.cur_exp + 1) * strategy.args.top_k

                if (prev_end > strategy.args.size) or (cur_end > strategy.args.size):
                    pass
                else:
                    cur_idx = (
                        slice(None), slice(None),
                        slice(cur_start, cur_end)) if strategy.args.use_prefix_tune_for_e_prompt else (
                        slice(None), slice(cur_start, cur_end))
                    prev_idx = (
                        slice(None), slice(None),
                        slice(prev_start, prev_end)) if strategy.args.use_prefix_tune_for_e_prompt else (
                        slice(None), slice(prev_start, prev_end))

                    with torch.no_grad():
                        if strategy.args.distributed:
                            strategy.model.module.e_prompt.prompt.grad.zero_()
                            strategy.model.module.e_prompt.prompt[cur_idx] = strategy.model.module.e_prompt.prompt[
                                prev_idx]
                            strategy.optimizer.param_groups[0]['params'] = strategy.model.module.parameters()
                        else:
                            strategy.model.e_prompt.prompt.grad.zero_()
                            strategy.model.e_prompt.prompt[cur_idx] = strategy.model.e_prompt.prompt[prev_idx]
                            strategy.optimizer.param_groups[0]['params'] = strategy.model.parameters()

        # Transfer previous learned prompt param keys to the new prompt
        if strategy.args.prompt_pool and strategy.args.shared_prompt_key:
            if self.cur_exp > 0:
                prev_start = (self.cur_exp - 1) * strategy.args.top_k
                prev_end = self.cur_exp * strategy.args.top_k

                cur_start = prev_end
                cur_end = (self.cur_exp + 1) * strategy.args.top_k

                with torch.no_grad():
                    if strategy.args.distributed:
                        strategy.model.module.e_prompt.prompt_key.grad.zero_()
                        strategy.model.module.e_prompt.prompt_key[cur_idx] = strategy.model.module.e_prompt.prompt_key[
                            prev_idx]
                        strategy.optimizer.param_groups[0]['params'] = strategy.model.module.parameters()
                    else:
                        strategy.model.e_prompt.prompt_key.grad.zero_()
                        strategy.model.e_prompt.prompt_key[cur_idx] = strategy.model.e_prompt.prompt_key[prev_idx]
                        strategy.optimizer.param_groups[0]['params'] = strategy.model.parameters()

        # Create new optimizer for each task to clear optimizer status
        if self.cur_exp > 0 and strategy.args.reinit_optimizer:
            strategy.optimizer = create_optimizer(strategy.args, strategy.model)

    def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        representation = []
        for batch_data in self.exp_x:
            with torch.no_grad():
                batch_representation = strategy.model(batch_data)['x'][:, 0]
                batch_representation = torch.nn.functional.normalize(batch_representation)
                representation.append(batch_representation)
        representation = torch.cat(tuple(representation), dim=0)
        self.exp_y = torch.cat(tuple(self.exp_y), dim=0)

        # 普通聚类
        for i in range(len(self.proto_labels)):
            proto = torch.tensor(self.protos[i]).to(strategy.device)
            proto_masked = self.random_mask(strategy, proto, 0.1)
            label = torch.tensor(self.proto_labels[i]).to(strategy.device)
            pre = strategy.model.head(proto.type(torch.float32))
            pre_mask = strategy.model.head(proto_masked.type(torch.float32))
            criterion = nn.MSELoss(reduce=True, reduction='sum')
            c = criterion(pre, pre_mask)
            prompt_loss = nn.CrossEntropyLoss()(pre, label) + c
            strategy.optimizer.zero_grad()
            prompt_loss.backward()
            strategy.optimizer.step()

        self.protoSave(representation)
        return

    def after_training_iteration(self, strategy, **kwargs):
        """
        Example callback: before backward.
        """

        benchmark = strategy.benchmark
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
            q[i] = 1 / num_seen_class[i]
        m = sum(q.values())
        S = {}
        for c in q:
            S[c] = q[c] / m
        return S

    def random_mask(self, strategy, x, mask_radio):
        w = x.shape[0]
        x1 = [mask_radio, 1 - mask_radio]
        mask = [1] * w
        for i in range(w):
            nums = np.random.choice([0, 1], size=1, p=x1)
            mask[i] *= nums[0]
        return x * torch.tensor(mask).to(strategy.device)

    def after_eval_iteration(
            self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:
        self.eval_exp_y.append(strategy.mb_y)
        self.eval_output.append(strategy.mb_output)

    # def after_eval_exp(
    #         self, strategy: Template, *args, **kwargs
    # ) -> CallbackResult:
    #     correct = defaultdict(int)
    #     total = defaultdict(int)
    #     acc_class_list = defaultdict(int)
    #     pre = torch.cat(tuple([torch.max(i, 1)[1] for i in self.eval_output])).cpu().numpy().tolist()
    #     self.eval_exp_y = torch.cat(tuple(self.eval_exp_y)).cpu().numpy().tolist()
    #     for i in range(strategy.benchmark.n_classes):
    #         correct[i] = 0
    #     for pre, label in zip(pre, self.eval_exp_y):
    #         if pre == label:
    #             correct[pre] += 1
    #         total[label] += 1
    #     for i in correct:
    #         acc_class_list[i] = correct[i] / total[i]
    #     print("每个类的准确率为")
    #     for i in acc_class_list:
    #         print(i, acc_class_list[i])
    #     fileName = "stream_1_50size.xlsx"
    #     self.acc_l.append(acc_class_list)
    #     workbook = xw.Workbook(fileName)  # 创建工作簿
    #     worksheet1 = workbook.add_worksheet("sheet1")  # 创建子表
    #     worksheet1.activate()  # 激活表
    #     i = 1  # 从第二行开始写入数据
    #     exp = self.count + 1
    #
    #     # print(exp)
    #     # exp_col = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I', 10: 'J', 11: 'K', 12: 'L',
    #     #            13: 'M', 14: 'N', 15: 'O', 16: 'P', 17: 'Q', 18: 'R', 19: 'S', 20: 'T', 21: 'U', 22: 'V', 23: 'W',
    #     #            24: 'X', 25: 'Y', 26: 'Z', 27: 'AA', 28: 'AB', 29: 'AC', 30: 'AD', 31: 'AE', 32: 'AF', 33: 'AG', 34: 'AH',
    #     #            35: 'AI', 36: 'AJ', 37: 'AK', 38: 'AL',
    #     #            39: 'AM', 40: 'AN', 41: 'AO', 42: 'AP', 43: 'AQ', 44: 'AR', 45: 'AS', 46: 'AT', 47: 'AU', 48: 'AV', 49: 'AW',
    #     #            50: 'AX'}
    #     if exp == 50:
    #         for j in range(len(self.acc_l[0])):
    #             insertData = [self.acc_l[i][j] for i in range(len(self.acc_l))]
    #             row = "A" + str(i)
    #             worksheet1.write_row(row, insertData)
    #             i += 1
    #         workbook.close()  # 关闭表
    #     self.count += 1
    #     pass
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
                correct[pre] += 1
            total[label] += 1

        for i in correct:
            acc_class_list[i] = correct[i] / total[i]
        print("每个类的准确率为")
        for i in acc_class_list:
            print(i, acc_class_list[i])


        """将loss和acc写入列表"""
        acc_experience = strategy.evaluator.metrics[2]._metric._mean_accuracy.summed / strategy.evaluator.metrics[
            2]._metric._mean_accuracy.weight
        loss_experience = strategy.evaluator.metrics[6]._metric._mean_loss.summed / strategy.evaluator.metrics[
            6]._metric._mean_loss.weight
        self.acc_experience_list.append(acc_experience)
        self.loss_experience_list.append(loss_experience)

        """将loss和acc列表保存至文件"""
        if self.count == 49:
            text_txt = 'acc_dual_s3.txt'
            with open(text_txt, 'a') as text:
                s = ''
                for i in range(len(self.acc_experience_list)):
                    s += str(self.acc_experience_list[i]) + ' ' if i != len(self.acc_experience_list) - 1 else str(
                        self.acc_experience_list[i]) + '\n'
                text.write(s)
            text_txt = 'loss_dual_s3.txt'
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
