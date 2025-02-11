import numpy as np
import torch
import numpy
from torch import nn


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
            self.mb_output, repre = self.forward()  # (64,100)，(64,512) 当前iteration
            self._after_forward(**kwargs)  # pass


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
        #     y.append(self.mb_y)
        #     k.append(repre)
        # k = tuple(k)
        # y = tuple(y)
        # exp_y = torch.cat(y, dim=0)
        # representation = torch.cat(k, dim=0)
        # return representation, exp_y


