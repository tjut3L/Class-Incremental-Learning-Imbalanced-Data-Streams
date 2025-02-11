import dill
from torchvision.transforms import transforms

from avalanche.benchmarks.datasets import CIFAR100
from avalanche.benchmarks.datasets import Car
from avalanche.benchmarks.datasets import TinyImagenet
from avalanche.benchmarks.datasets import CUB200
import numpy as np
from .cir_benchmark import generate_benchmark
from avalanche.benchmarks.datasets import MiniImageNetDataset
from avalanche.benchmarks.datasets import Miniimagenet
from avalanche.benchmarks.datasets import ImagenetSubset

# from avalanche.benchmarks.datasets import
import torch

'''
[27, 35, 29, 26, 20, 32, 21, 44, 48, 18, 39, 26, 28, 46,  3,  4,  0,
       41, 38, 43, 48, 39, 22, 38,  5, 31,  7, 47, 25, 20, 12, 38, 22, 28,
        0, 30, 30, 30, 47, 33, 17, 21, 34,  2, 33, 33, 10,  6, 15, 17, 28,
       21, 49,  4, 10,  7, 32, 12, 23, 11,  7,  5, 32,  6,  9, 18, 40,  4,
       41,  4, 48, 23, 48, 29, 36,  1, 13,  5, 14,  5, 15, 20,  3, 34, 28,
       13, 25,  4, 28, 46, 15, 33,  6, 35, 14,  8, 29,  0, 41,  0]


'''

val_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5071, 0.4867, 0.4408),
    #                      (0.2675, 0.2565, 0.2761)),
    # transforms.Normalize(mean = (0.485, 0.456, 0.406),
    #                         std =  (0.229, 0.224, 0.225)),   #imagenet
    transforms.Normalize((0.5071, 0.4867, 0.4408),
                         (0.2675, 0.2565, 0.2761)),  #cifar100
    # transforms.RandomResizedCrop((32,32),scale=(0.8,1.0),ratio=(1.0,1.0))
    transforms.RandomResizedCrop((224,224),scale=(0.8,1.0),ratio=(1.0,1.0))
])


def get_cifar_based_benchmark(scenario_config, seed, dataset):
    # Download dataset if not available
    if dataset == "TinyImageNet":
        test_set = TinyImagenet(root="/home/HDD2/tjut_zhanghaiyang/hanlei", train=False,
                            transform=val_transform, download=True)
        
        train_set = TinyImagenet(root="/home/HDD2/tjut_zhanghaiyang/hanlei", train=True,
                            transform=val_transform, download=True)
        num_classes = len(train_set.id2label)
    elif dataset == "CUB200":
        test_set = CUB200(root="./data/datasets", train=False,
                            transform=val_transform, download=True)
        
        train_set = CUB200(root="./data/datasets", train=True,
                            transform=val_transform, download=True)
    elif dataset == "CIFAR100":
        test_set = CIFAR100(root="/home/HDD2/tjut_zhanghaiyang/datasets/CIFAR100", train=False,
                            transform=val_transform, download=True)

        train_set = CIFAR100(root="/home/HDD2/tjut_zhanghaiyang/datasets/CIFAR100", train=True,
                            transform=val_transform, download=True)
        num_classes = len(train_set.classes)
    num_samples = len(train_set.data)
    label = train_set.targets

    cls_positive = [[] for i in range(num_classes)]
    for i in range(num_samples):
        cls_positive[label[i]].append(i)
    cls_positive = [np.asarray(cls_positive[i]) for i in range(num_classes)]
    # print(train_set.data)
    # print(train_set.targets)
    # test_set = ImagenetSubset(root_dir="/home/HDD2/jskj_taozhe/imagenet", train=False,
    #                     transform=val_transform)

    # train_set = ImagenetSubset(root_dir="/home/HDD2/jskj_taozhe/imagenet", train=True,
    #                     transform=val_transform)

    # test_set = MiniImageNetDataset(imagenet_path='/home/HDD2/jskj_taozhe/imagenet',split='train')
    # train_set = MiniImageNetDataset(imagenet_path='/home/HDD2/jskj_taozhe/imagenet',split='test')

    # test_set = Car(root_dir="/home/HDD2/jskj_taozhe/Cars", train=False,
    #                     transform=val_transform)
    #
    # train_set = Car(root_dir="/home/HDD2/jskj_taozhe/Cars", train=True,
    #                     transform=val_transform)

    # test_set = Miniimagenet(root_dir="/home/HDD2/jskj_taozhe/food-101", train=False,
    #                     transform=val_transform)

    # train_set = Miniimagenet(root_dir="/home/HDD2/jskj_taozhe/food-101", train=True,
    #                     transform=val_transform)

    # test_set = Miniimagenet(root_dir="/home/HDD2/jskj_taozhe/mini-imagenet", train=False,
    #                     transform=val_transform)

    # train_set = Miniimagenet(root_dir="/home/HDD2/jskj_taozhe/mini-imagenet", train=True,
    #                     transform=val_transform)

    # Load challenge datasets
    # print('============================================================')
    # with open("/home/tjut_zhaoyishuo/clvision-challenge-23/dataallenge_train_set.pkl", "rb") as pkl_file:
    #     train_set = dill.load(pkl_file)
    #     # print(train_set)
    #
    # with open("/home/tjut_zhaoyishuo/clvision-challenge-2023-main/dataallenge_test_set.pkl", "rb") as pkl_file:
    #     test_set = dill.load(pkl_file)
    # print(test_set)

    """
    'scenario_table' 100*50 每个类在哪个任务出现
    'n_samples_table', 100*50  每一类数据出现的示例个数
     'n_classes',  类总数
     'n_e',        任务数
     'first_occurrences',   每个类第一次出现的任务索引
     'indices_per_class'    每个类的数据的索引
    """
    with open(f"/home/tjut_zhanghaiyang/paper-code/aa/zhao/cl-CIR-1/scenario_configs/{scenario_config}",
              "rb") as pkl_file:
        scenario_config = dill.load(pkl_file)
    # print(scenario_config)
    # return scenario_config

    # # # Benchmarks
    indices = scenario_config["indices_per_class"]
    for i in indices:
        scenario_config["indices_per_class"][i] = torch.tensor(cls_positive[i])
    benchmark = generate_benchmark(seed=seed, train_set=train_set,
                                   test_set=test_set, **scenario_config)
    return benchmark


# get_cifar_based_benchmark('imagenet_sub_3.pkl', 2)
# k = se['n_samples_table']
# m = se['first_occurrences']
# print(m)
# print(k)
# sum1 = 0
# sum2 = 0
# for i in range(50):
#     sum1 += 99-m[i]
#     p = [j for j in k[i] if j != 0]
#     sum2+=len(p)-1
#
# print(sum2/sum1)

# loadData = np.load('./pred_config_s2_run1.npy')
# print(loadData)
# output_name = f"./pred_config_s1_run1.npy"
# print(output_name)
# np.save(output_name, [1, 2.3])
__al__ = ["get_cifar_based_benchmark"]
