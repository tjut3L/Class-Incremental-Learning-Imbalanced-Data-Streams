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
from avalanche.models.bic_model import BiasLayer


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
        self.bias_layer = {}
        self.proto_labels = []
        self.protos = []
        self.exp_y = []
        self.exp_x = []
        self.new_S = None
        self.cur_exp = 0
        self.n_classes = 100
        self.old_prompt = None
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

    def after_forward(
            self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:
        if strategy.mb_output is not None:
            for t in self.bias_layer.keys():
                strategy.mb_output = self.bias_layer[t](strategy.mb_output.unsqueeze(0)).reshape(-1)

    def after_eval_forward(self, strategy, **kwargs):
        if strategy.mb_output is not None:
            for t in self.bias_layer.keys():
                strategy.mb_output = self.bias_layer[t](strategy.mb_output.unsqueeze(0))

    def protoSave(self, feature):  # feature (2000,768)
        features = []
        labels = []
        for i in range(feature.shape[0]):
            features.append(feature[i].cpu().numpy())
            labels.append(int(self.exp_y[i]))

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
        return

    def freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False
        return model

    def un_freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = True
        return model

    def _get_optimizer(self, strategy):
        """Returns the optimizer"""
        for name, parameter in strategy.model.named_parameters():
            if 'prompt' not in name and 'head' not in name:
                parameter.requires_grad = False
        params = list(strategy.model.head.parameters())
        if strategy.model.e_prompt:
            params += strategy.model.e_prompt.parameters()
        if strategy.model.g_prompt:
            params += strategy.model.g_prompt.parameters()
        enabled = set()
        for i in self.bias_layer:
            params += self.bias_layer[i].parameters()
            # double check, 检查需要更新的参数列表
            enabled.add("bia_layer_" + str(i))
        for name, param in strategy.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        return torch.optim.Adam(params, lr=0.001)

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

        # BIC
        task_id = strategy.experience.current_experience
        if task_id not in self.bias_layer:
            self.bias_layer[task_id] = BiasLayer(
                strategy.device,
                list(strategy.adapted_dataset.targets.uniques)
            )
            strategy.optimizer = self._get_optimizer(strategy)
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
        self.protoSave(representation)
        task_id = strategy.experience.current_experience

        bic_optimizer = torch.optim.Adam(
            self.bias_layer[task_id].parameters(),
            lr=0.001)
        # 加权聚类
        loss_protoaug = torch.tensor(0.0).to(strategy.device)
        # for i in range(len(self.proto_labels)):
        #     proto = torch.tensor(self.protos[i]).to(strategy.device)
        #     label = torch.tensor(self.proto_labels[i]).to(strategy.device)
        #     alpha = 100 * torch.tensor(self.new_S[self.proto_labels[i]]).to(strategy.device)
        #     pre = strategy.model.head(proto)
        #     prompt_loss = nn.CrossEntropyLoss()(pre, label)
        #     loss_protoaug += alpha * prompt_loss
        #     # print("prompt_loss: ", prompt_loss)
        #
        # strategy.optimizer.zero_grad()
        # loss_protoaug.backward()
        # strategy.optimizer.step()



        for i in range(len(self.proto_labels)):
            proto = torch.tensor(self.protos[i]).to(strategy.device)
            label = torch.tensor(self.proto_labels[i]).to(strategy.device)
            # proto = proto.unsqueeze(0)
            # pre = strategy.model.head(proto).reshape(-1)
            # prompt_loss = nn.CrossEntropyLoss()(pre/0.5, label)
            # strategy.optimizer.zero_grad()

            pre = strategy.model.head(proto)
            for t in self.bias_layer.keys():
                pre = self.bias_layer[t](pre)
            prompt_loss = nn.CrossEntropyLoss()(pre, label.reshape(-1))
            strategy.optimizer.zero_grad()
            prompt_loss.backward()
            strategy.optimizer.step()
        #bias_loss
        # loss_protoaug = 0.1 * ((self.bias_layer[task_id].beta.sum()
        #                         ** 2) / 2)
        # bic_optimizer.zero_grad()
        # loss_protoaug.backward()
        # bic_optimizer.step()

        return

    # def after_training_epoch(
    #     self, strategy: Template, *args, **kwargs
    # ) -> CallbackResult:
    #         prompt_loss = torch.dist(self.old_prompt, strategy.model.e_prompt.prompt, p=2)
    #         print("prompt_loss: ",prompt_loss)
    #         strategy.optimizer.zero_grad()
    #         prompt_loss.backward()
    #         strategy.optimizer.step()
    #
    #         pass

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

    def after_eval_iteration(
            self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:
        pass

    def after_eval_exp(
            self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:
        pass
