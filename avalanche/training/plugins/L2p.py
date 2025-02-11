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


class L2PPlugin(SupervisedPlugin):
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
        self.acc_l = []
        self.exp_y = []
        self.exp_x = []
        self.count = 0
        self.protos = []
        self.cur_exp = 0
        self.n_classes = 100


        super().__init__()
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
        with torch.no_grad():
            if strategy.original_model is not None:
                output = strategy.original_model(strategy.mb_x)
                strategy.cls_features = output['pre_logits']
            else:
                strategy.cls_features = None
        pass


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
        self.exp_y = []
        self.exp_x = []
        self.eval_exp_y = []
        self.eval_exp_x = []
        self.eval_output = []
        self.eval_output = []
        self.cur_exp = strategy.experience.current_experience
        if strategy.args.prompt_pool and strategy.args.shared_prompt_pool:
            if self.cur_exp > 0:
                prev_start = (self.cur_exp - 1) * strategy.args.top_k
                prev_end = self.cur_exp * strategy.args.top_k

                cur_start = prev_end
                cur_end = (self.cur_exp + 1) * strategy.args.top_k

                if (prev_end > strategy.args.size) or (cur_end > strategy.args.size):
                    pass
                else:
                    cur_idx = (slice(cur_start, cur_end))
                    prev_idx = (slice(prev_start, prev_end))

                    with torch.no_grad():
                        if strategy.args.distributed:
                            strategy.model.module.prompt.prompt.grad.zero_()
                            strategy.model.module.prompt.prompt[cur_idx] = strategy.model.module.prompt.prompt[prev_idx]
                            strategy.optimizer.param_groups[0]['params'] = strategy.model.module.parameters()
                        else:
                            strategy.model.prompt.prompt.grad.zero_()
                            strategy.model.prompt.prompt[cur_idx] = strategy.model.prompt.prompt[prev_idx]
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
                        strategy.model.module.prompt.prompt_key.grad.zero_()
                        strategy.model.module.prompt.prompt_key[cur_idx] = strategy.model.module.prompt.prompt_key[prev_idx]
                        strategy.optimizer.param_groups[0]['params'] = strategy.model.module.parameters()
                    else:
                        strategy.model.prompt.prompt_key.grad.zero_()
                        strategy.model.prompt.prompt_key[cur_idx] = strategy.model.prompt.prompt_key[prev_idx]
                        strategy.optimizer.param_groups[0]['params'] = strategy.model.parameters()

        # Create new optimizer for each task to clear optimizer status
        if self.cur_exp > 0 and strategy.args.reinit_optimizer:
            strategy.optimizer = create_optimizer(strategy.args, strategy.model)

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

    def random_mask(self, strategy, x, mask_radio):
        w = x.shape[0]
        x1 = [mask_radio, 1 - mask_radio]
        mask = [1] * w
        for i in range(w):
            nums = np.random.choice([0, 1], size=1, p=x1)
            mask[i] *= nums[0]
        return x * torch.tensor(mask).to(strategy.device)


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
            prompt_loss = nn.CrossEntropyLoss()(pre, label)
            strategy.optimizer.zero_grad()
            prompt_loss.backward()
            strategy.optimizer.step()

        self.protoSave(representation)
        return

    def after_training_epoch(
        self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:
        # acc1, acc5 = accuracy(strategy.mb_output, strategy.mb_y, topk=(1, 5))
        # print("ACC@1:",acc1)
        # print("ACC@5:",acc5)
        pass

    def after_training_iteration(self, strategy, **kwargs):
        """
        Example callback: before backward.
        """
        benchmark = strategy.benchmark
        if strategy.clock.train_exp_epochs == 0:
            self.exp_x.append(strategy.mb_x)
            self.exp_y.append(strategy.mb_y)
        pass


    def after_eval_iteration(
            self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:
        self.eval_exp_y.append(strategy.mb_y)
        self.eval_output.append(strategy.mb_output)
        pass

    def after_eval_exp(
            self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:
        pass

