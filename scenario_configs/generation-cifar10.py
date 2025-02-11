import copy
import pickle
import random
from collections import defaultdict

import dill
import numpy as np
import torch

'''
    'scenario_table' 100*50 每个类在哪个任务出现
    'n_samples_table', 100*50  每一类数据出现的示例个数
     'n_classes',  类总数
     'n_e',        任务数
     'first_occurrences',   每个类第一次出现的任务索引
     'indices_per_class'    每个类的数据的索引
'''

'''
cub-200 分割数据集的index
'''

image_labels = {}
data_split = defaultdict(list)

with open("/home/HDD2/jskj_taozhe/imagenet/train.txt", 'r') as f:
    q = f.readlines()
    index = 0
    for k in q:
        p = k.split(' ')
        if int(p[1]) < 100:
            data_split[int(p[1])].append(index)
        index += 1

p_fir = 0.1 # 0.5 0.2 0.02
num_samples_per_task = 2000

n_e = 50
n_classes = 20
p_re = [0] * n_classes
first_occurrences = [-1] * n_classes
n_samples_table = [[0] * n_e for _ in range(n_classes)]
indices_per_class = {}
scenario_table = [[0] * n_e for _ in range(n_classes)]

# t = defaultdict(list)
# for i in image_labels:
#     if data_split[i] == 1:
#         t[image_labels[i]-1].append(i)
#
# for i in t:
#     indices_per_class[i] = torch.tensor(t[i])
for i in data_split:
    print(len(data_split[i]))
    indices_per_class[i] = torch.tensor(data_split[i])

n_class_per_task = {}
m = [i for i in range(n_classes)]
seen_class = []
task_id = 0
for i in range(n_classes):
    r = random.uniform(0, 1)
    p_re[i] = r
# for i in range(len(p_re)):
#     if p_re[i]>0.5:
#         r = random.uniform(0, 0.5)
#         p_re[i] = r
while m and task_id < 50:
    nums = np.random.choice([0, 1], size=len(m), p=[1 - p_fir, p_fir])
    k = []
    delete_index = []
    new_m = []
    for i in range(n_classes):
        if i in seen_class:
            nums1 = np.random.choice([0, 1], size=1, p=[1 - p_re[i], p_re[i]])
            if nums1[0] == 1:
                k.append(i)
    for i in range(len(nums)):
        if nums[i] == 1:
            k.append(m[i])
            delete_index.append(i)
            first_occurrences[m[i]] = task_id
            seen_class.append(m[i])
        else:
            new_m.append(m[i])

    m = copy.deepcopy(new_m)
    n_class_per_task[task_id] = k
    task_id += 1
    task_id = task_id % 50

for i in n_class_per_task:
    n_samples = num_samples_per_task // len(n_class_per_task[i])
    for j in n_class_per_task[i]:
        n_samples_table[j][i] = n_samples
        scenario_table[j][i] = 1

# for i in range(200):
#     p = torch.randperm(500)
#     m = []
#     for j in p:
#         q = j+i*500
#         m.append(q.item())
#     indices_per_class[i] = torch.tensor(m)


m_save = {}
m_save["first_occurrences"] = first_occurrences
m_save["n_samples_table"] = torch.tensor(n_samples_table)
m_save["n_e"] = n_e
m_save["n_classes"] = n_classes
m_save["indices_per_class"] = indices_per_class
m_save["scenario_table"] = torch.tensor(scenario_table)

f = open('cifar10_1.pkl', 'wb')
pickle.dump(m_save, f)

print(first_occurrences)
print(n_class_per_task)
