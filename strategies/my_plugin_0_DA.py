import pickle
from typing import Optional, TYPE_CHECKING, List
from torchvision import transforms
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


class MyPluginDA(SupervisedPlugin):
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
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.batch_size_mem = batch_size_mem
        self.redius = 0
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


    def before_training_exp(
            self,
            strategy: "SupervisedTemplate",
            num_workers: int = 0,
            shuffle: bool = True,
            **kwargs
    ):
        if strategy.experience.current_experience == 10:
            strategy.train_epochs = 50
        if strategy.experience.current_experience == 20:
            strategy.train_epochs = 40
        if strategy.experience.current_experience == 30:
            strategy.train_epochs = 30
        if strategy.experience.current_experience == 40:
            strategy.train_epochs = 20


        # strategy.train_epochs = 1

    def before_training_iteration(
            self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:
        # pass
        # strategy.mbatch[0] = torch.stack([strategy.mbatch[0], transforms.RandomHorizontalFlip(p=1)(strategy.mbatch[0]),transforms.RandomVerticalFlip(p=1)(strategy.mbatch[0])], 1)
        strategy.mbatch[0] = torch.stack([strategy.mbatch[0], transforms.RandomHorizontalFlip(p=1)(strategy.mbatch[0])], 1)
        strategy.mbatch[0] = strategy.mbatch[0].view(-1, 3, 256, 256)
        strategy.mbatch[1] = torch.stack([strategy.mbatch[1] for k in range(2)], 1).view(-1)



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
        acc_experience = strategy.evaluator.metrics[2]._metric._mean_accuracy.summed / strategy.evaluator.metrics[2]._metric._mean_accuracy.weight
        loss_experience = strategy.evaluator.metrics[6]._metric._mean_loss.summed / strategy.evaluator.metrics[6]._metric._mean_loss.weight
        self.acc_experience_list.append(acc_experience)
        self.loss_experience_list.append(loss_experience)

        """将loss和acc列表保存至文件"""
        if self.count == 49:
            text_txt = 'acc_sub_3_lwf_da-n.txt'
            with open(text_txt, 'a') as text:
                    s = ''
                    for i in range(len(self.acc_experience_list)):
                        s += str(self.acc_experience_list[i]) + ' ' if i != len(self.acc_experience_list) - 1 else str(self.acc_experience_list[i]) + '\n'
                    text.write(s)
            text_txt = 'loss_sub_3_lwf_da-n.txt'
            with open(text_txt, 'a') as text:
                    s = ''
                    for i in range(len(self.loss_experience_list)):
                        s += str(self.loss_experience_list[i]) + ' ' if i != len(self.loss_experience_list) - 1 else str(self.loss_experience_list[i]) + '\n'
                    text.write(s)
            """将训练好的model保存至文件"""
            # with open('tiny_cub_2_lwf_da_net.pkl', 'wb') as f:
            #     pickle.dump(strategy.model, f)


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
