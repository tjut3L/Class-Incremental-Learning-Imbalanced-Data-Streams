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


class MyPlugin0(SupervisedPlugin):
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
        self.cll = defaultdict(set)
        self.eval_exp_x = []
        self.count_e = defaultdict(int)
        self.eval_exp_y = []
        self.count = 0
        if storage_policy is not None:  # Use other storage policy
            self.storage_policy = storage_policy
            assert storage_policy.max_size == self.mem_size
        else:  # Default
            self.storage_policy = ERBuffer(
                max_size=self.mem_size,
            )
        self.count += 1


    def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        if strategy.experience.current_experience == 49:
            print(self.count_e)
            text_txt = "num_examplar2.txt"
            with open(text_txt, 'a') as text:
                s = ''
                for i in self.count_e:
                    s += str(self.count_e[i]) + ' '
                text.write(s)
            print(self.cll)


    def after_training_iteration(self, strategy, **kwargs):
        """
        Example callback: before backward.
        """

        for i in strategy.mb_y.cpu().numpy().tolist():
            self.count_e[i] += 1
            self.cll[strategy.experience.current_experience].add(i)
        pass