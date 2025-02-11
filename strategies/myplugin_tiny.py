import pickle
from collections import defaultdict
from typing import Optional, TYPE_CHECKING, List

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from avalanche.core import Template, CallbackResult
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.storage_policy import (
    ExemplarsBuffer,
    ExperienceBalancedBuffer, HerdingSelectionStrategy, ERBuffer
)


if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate


class MyPluginTiny(SupervisedPlugin):
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
        self.eval_y = []
        self.eval_x = []
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.batch_size_mem = batch_size_mem
        self.redius = 0
        self.acc_experience_list = []
        self.loss_experience_list = []
        self.eval_exp_x = []
        self.eval_exp_y = []
        self.count = 0
        """
        'scenario_table' 100*50 每个类在哪个任务出现
        'n_samples_table', 100*50  每一类数据出现的示例个数
         'n_classes',  类总数
         'n_e',        任务数
         'first_occurrences',   每个类第一次出现的任务索引
         'indices_per_class'    每个类的数据的索引
        """
        self.first_occurence = defaultdict(int)
        self.first_occurence.setdefault(-1)
        self.per_class_exp = defaultdict(set)
        self.n_samples_table = defaultdict(int)
        if storage_policy is not None:  # Use other storage policy
            self.storage_policy = storage_policy
            assert storage_policy.max_size == self.mem_size
        else:  # Default
            self.storage_policy = ERBuffer(
                max_size=self.mem_size,
            )

    def before_backward(
        self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:
        for i in strategy.mb_y:
            if self.first_occurence[i] == -1:
                self.first_occurence[i] = strategy.experience.current_experience
            self.per_class_exp[strategy.experience.current_experience].add(i)
            self.n_samples_table[i]+=1
    def before_training_exp(
            self,
            strategy: "SupervisedTemplate",
            num_workers: int = 0,
            shuffle: bool = True,
            **kwargs
    ):
        # if strategy.experience.current_experience == 10:
        #     strategy.train_epochs = 50
        # if strategy.experience.current_experience == 20:
        #     strategy.train_epochs = 40
        # if strategy.experience.current_experience == 30:
        #     strategy.train_epochs = 30
        # if strategy.experience.current_experience == 40:
        #     strategy.train_epochs = 20

        pass
        strategy.train_epochs = 1

    def after_training_exp(self, strategy: Template, *args, **kwargs):
        self.eval_y = []
        pass

    def after_eval_iteration(
        self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:
        if self.count == 0:
            self.eval_exp_x.append(strategy.mb_x)
            self.eval_exp_y.append(strategy.mb_y)



    def after_training_iteration(
        self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:
        if strategy.clock.train_exp_epochs == 0:
            for i in strategy.mb_y:
                if i not in self.eval_y:
                    self.eval_y.append(i)



    def after_eval_exp(
        self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:

        """将loss和acc写入列表"""
        acc_experience = strategy.evaluator.metrics[2]._metric._mean_accuracy.summed / strategy.evaluator.metrics[2]._metric._mean_accuracy.weight
        loss_experience = strategy.evaluator.metrics[6]._metric._mean_loss.summed / strategy.evaluator.metrics[6]._metric._mean_loss.weight
        self.acc_experience_list.append(acc_experience)
        self.loss_experience_list.append(loss_experience)

        """将loss和acc列表保存至文件"""
        if self.count == 49:
            text_txt = 'acc_s3_ewc.txt'
            with open(text_txt, 'a') as text:
                    s = ''
                    for i in range(len(self.acc_experience_list)):
                        s += str(self.acc_experience_list[i]) + ' ' if i != len(self.acc_experience_list) - 1 else str(self.acc_experience_list[i]) + '\n'
                    text.write(s)
            text_txt = 'loss_s3_ewc.txt'
            with open(text_txt, 'a') as text:
                    s = ''
                    for i in range(len(self.loss_experience_list)):
                        s += str(self.loss_experience_list[i]) + ' ' if i != len(self.loss_experience_list) - 1 else str(self.loss_experience_list[i]) + '\n'
                    text.write(s)
            """将训练好的model保存至文件"""
            with open('ewc_model_s3.pkl', 'wb') as f:
                pickle.dump(strategy.model, f)
            print('第一次出现：',self.first_occurence)
            print("每个任务出现的类：",self.per_class_exp)
            # print(self.n_samples_table)

        """最后一个任务结束后对test_set进行特征提取"""
        # if self.count == 49:
        #     with torch.no_grad():
        #         representation = []
        #         for batch_data in self.eval_exp_x:
        #             batch_representation = strategy.model(batch_data)[1]
        #             batch_representation = torch.nn.functional.normalize(batch_representation)
        #             representation.append(batch_representation.cpu().numpy())
        #         representation = np.concatenate(representation)
        #         # representation = torch.cat(tuple(representation), dim=0)
        #         # labels = torch.cat(tuple(self.eval_exp_y), dim=0)
        #
        #         kmeans = KMeans(n_clusters=10)
        #         kmeans_labels = kmeans.fit_predict(representation)
        #         """可视化"""
        #         tsne = TSNE(n_components=2, random_state=0)
        #         X_tsne = tsne.fit_transform(representation)
        #         plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=1, c=kmeans_labels)
        #         plt.xlabel("x")
        #         plt.ylabel("y")
        #         plt.title("LWF")
        #         plt.show()

        self.count += 1
