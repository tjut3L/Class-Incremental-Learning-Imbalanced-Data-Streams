"""
The :py:mod:`benchmarks` module provides a set of utilities that can be used for
handling and generating your continual learning data stream. In the
:py:mod:`datasets` module, basic PyTorch Datasets are provided. In the
:py:mod:`classic` module instead, classic benchmarks (already proposed in the
CL literature) generated from the datasets are provided. Finally,
in :py:mod:`generators` basic utilities to generate new benchmarks on-the-fly
are made available.
"""

from .scenarios import *
from .generators import *
from .classic import *
#
# import dill
#
#
# from avalanche.benchmarks.datasets import CIFAR100
#
# from cir_benchmark import generate_benchmark
# #
# #
# import torch
# '''
# [27, 35, 29, 26, 20, 32, 21, 44, 48, 18, 39, 26, 28, 46,  3,  4,  0,
#        41, 38, 43, 48, 39, 22, 38,  5, 31,  7, 47, 25, 20, 12, 38, 22, 28,
#         0, 30, 30, 30, 47, 33, 17, 21, 34,  2, 33, 33, 10,  6, 15, 17, 28,
#        21, 49,  4, 10,  7, 32, 12, 23, 11,  7,  5, 32,  6,  9, 18, 40,  4,
#        41,  4, 48, 23, 48, 29, 36,  1, 13,  5, 14,  5, 15, 20,  3, 34, 28,
#        13, 25,  4, 28, 46, 15, 33,  6, 35, 14,  8, 29,  0, 41,  0]
#
#
# '''
# def get_cifar_based_benchmark(scenario_config, seed):
#     # Download dataset if not available
#     _ = CIFAR100(root="./data/datasets", train=True,
#                      transform=None, download=True)
#     _ = CIFAR100(root="./data/datasets", train=False,
#                      transform=None, download=True)
#
#     # Load challenge datasets
#     with open("/home/tjut_zhaoyishuo/clvision-challenge-2023-main/data/challenge_train_set.pkl", "rb") as pkl_file:
#         train_set = dill.load(pkl_file)
#         # print(train_set)
#
#     with open("/home/tjut_zhaoyishuo/clvision-challenge-2023-main/data/challenge_test_set.pkl", "rb") as pkl_file:
#         test_set = dill.load(pkl_file)
#         # print(test_set)
#
#     """
#     'scenario_table' 100*50 每个类在哪个任务出现
#     'n_samples_table', 100*50  每一类数据出现的示例个数
#      'n_classes',  类总数
#      'n_e',        任务数
#      'first_occurrences',   每个类第一次出现的任务索引
#      'indices_per_class'    每个类的数据的索引
#     """
#     with open(f"/home/tjut_zhaoyishuo/clvision-challenge-2023-main/scenario_configs/{scenario_config}", "rb") as pkl_file:
#         scenario_config = dill.load(pkl_file)
#     print(scenario_config)
#     # # Benchmarks
#     benchmark = generate_benchmark(seed=seed, train_set=train_set,
#                                    test_set=test_set, **scenario_config)
#     print(benchmark)
#     return benchmark
#
#
# # get_cifar_based_benchmark('config_s2.pkl',2)
# __al__ = ["get_cifar_based_benchmark"]
