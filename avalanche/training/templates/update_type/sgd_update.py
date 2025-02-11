import numpy as np
import torch
import numpy
from torch import nn
from torch.distributions import normal
import torch.nn.functional as F


def focal_loss_new(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)  # 目标类概率
    loss = (1 - p.detach()) ** gamma * input_values
    return loss.mean()


class GCLLoss(nn.Module):

    def __init__(self, cls_num_list, m=0.5, weight=None, s=30, train_cls=False, noise_mul=1., gamma=0.):
        super(GCLLoss, self).__init__()
        cls_list = torch.cuda.FloatTensor(cls_num_list)
        m_list = torch.log(cls_list)
        m_list = m_list.max() - m_list
        self.m_list = m_list
        assert s > 0
        self.m = m
        self.s = s
        self.weight = weight
        self.simpler = normal.Normal(0, 1 / 3)
        self.train_cls = train_cls
        self.noise_mul = noise_mul
        self.gamma = gamma

    def forward(self, cosine, target):
        index = torch.zeros_like(cosine, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        noise = self.simpler.sample(cosine.shape).clamp(-1, 1).to(
            cosine.device)  # self.scale(torch.randn(cosine.shape).to(cosine.device))

        # cosine = cosine - self.noise_mul * noise/self.m_list.max() *self.m_list
        cosine = cosine - self.noise_mul * noise.abs() / self.m_list.max() * self.m_list
        output = torch.where(index, cosine - self.m, cosine)
        if self.train_cls:
            return focal_loss_new(F.cross_entropy(self.s * output, target, reduction='none', weight=self.weight),
                                  self.gamma)
        else:
            return F.cross_entropy(self.s * output, target, weight=self.weight)


class SGDUpdate:
    def training_epoch(self, **kwargs):
        """Training epoch.

        :param kwargs:
        :return:
        """
        k = []
        y = []
        for self.mbatch in self.dataloader:
            if self._stop_training:
                break

            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)  # 数据增强

            self.optimizer.zero_grad()
            self.loss = 0

            # Forward
            self._before_forward(**kwargs)  # pass
            self.output_list = self.forward()  # (64,100)，(64,512) 当前iteration

            self._after_forward(**kwargs)  # pass
            self.mb_output = self.output_list[0]

            # Loss
            self.loss += nn.CrossEntropyLoss()(self.mb_output, self.mb_y)
            # Backward+self._criterion()
            self._before_backward(**kwargs)  #
            self.backward()
            self._after_backward(**kwargs)  # pass

            # Optimization step
            self._before_update(**kwargs)  # pass
            self.optimizer_step()
            self._after_update(**kwargs)  # pass

            self._after_training_iteration(**kwargs)  # 将当前batch的数据和label加入到exp_x和exp_y中
# import numpy as np
# import torch
# import numpy
# from torch import nn


# class SGDUpdate:
#     def training_epoch(self, **kwargs):
#         """Training epoch.
#
#         :param kwargs:
#         :return:
#         """
#         k = []
#         y = []
#         for self.mbatch in self.dataloader:
#             if self._stop_training:
#                 break
#
#             self._unpack_minibatch()
#             self._before_training_iteration(**kwargs)  # 数据增强
#
#             self.optimizer.zero_grad()
#             self.loss = 0
#
#             # Forward
#             self._before_forward(**kwargs)  # pass
#             self.mb_output,pre = self.forward()  # (64,100)，(64,512) 当前iteration
#             self._after_forward(**kwargs)  # pass
#
#
#             # Loss
#             self.loss += nn.CrossEntropyLoss()(self.mb_output, self.mb_y)
#             # Backward+self._criterion()
#             self._before_backward(**kwargs)  #
#             self.backward()
#             self._after_backward(**kwargs)  # p
#             # ass
#
#
#             # Optimization step
#             self._before_update(**kwargs)  # pass
#             self.optimizer_step()
#             self._after_update(**kwargs)  # pass
#
#             self._after_training_iteration(**kwargs)  # 将当前batch的数据和label加入到exp_x和exp_y中