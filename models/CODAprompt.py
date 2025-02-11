import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from argparse import ArgumentParser
from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset


class Appr(Inc_Learning_Appr):
    """Class implementing the finetuning baseline"""

    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5,
                 clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False, eval_on_train=False,
                 nc_first=None, num_tasks=10, template="", classes_names=None, classes_array=None, class_order=None,
                 logger=None, exemplars_dataset=None, all_outputs=False, _multiple_gpus=None, DW=False):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, nc_first, num_tasks,
                                   template, classes_names, classes_array, class_order, logger,
                                   exemplars_dataset)
        self.all_out = all_outputs
        self.criterion_fn = nn.CrossEntropyLoss(reduction='none')
        self._multiple_gpus = _multiple_gpus
        self.DW = DW
        # highest class index from past task
        self.last_valid_out_dim = 0
        # highest class index from current task
        self.valid_out_dim = 0
        self.add_dim = 0

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--all-outputs', action='store_true', required=False,
                            help='Allow all weights related to all outputs to be modified (default=%(default)s)')
        parser.add_argument('--_multiple_gpus', default=[2], type=list, required=False,
                            help='using multiple gpus')
        parser.add_argument('--DW', action='store_true', required=False,
                            help='dataset balancing')
        return parser.parse_known_args(args)

    def _get_optimizer(self, t):
        """Returns the optimizer"""
        for name, parameter in self.model.named_parameters():
            if 'prompt' not in name and 'head' not in name:
                parameter.requires_grad = False
        params = list(self.model.prompt.parameters()) + \
                 list(self.model.head.parameters())
        # double check, 检查需要更新的参数列表
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        return torch.optim.Adam(params, lr=self.lr, weight_decay=self.wd, betas=(self.momentum, 0.999))

    def pre_train_process(self, t, trn_loader):
        """Runs before training all epochs of the task (before the train session)"""
        pass

    def freeze(self, model):
        for p in model.parameters():
            p.requires_grad = False

    def train_loop(self, t, trn_loader, val_loader, num_old, num_cur):
        """Contains the epochs loop"""
        # add exemplars to train_loader
        if len(self.exemplars_dataset) > 0 and t > 0:
            trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory)

        # FINETUNING TRAINING -- contains the epochs loop
        self.optimizer = self._get_optimizer(t)
        scheduler = CosineSchedule(self.optimizer, K=self.nepochs)
        # dtype = self.model.dtype
        if t > 0:
            try:
                if self.model.module.prompt is not None:
                    self.model.module.prompt.process_task_count()
            except:
                if self.model.prompt is not None:
                    self.model.prompt.process_task_count()
        if self._multiple_gpus[0] >= 0:
            self.cuda()
        trainable_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(trainable_num)
        self.add_dim = num_cur - num_old
        self.add_valid_output_dim(self.add_dim)
        self.data_weighting()
        for e in range(self.nepochs):
            # Train
            clock0 = time.time()
            self.model.train()
            losses = 0.0
            for images, true_labels, captions, targets in trn_loader:
                # Forward current model
                images, targets = images.cuda(), targets.cuda()
                # images.to(self.model.feat.dtype)
                loss, output = self.update_model(images, targets)
                # loss, output = self.update_model(images.to(self.model.dtype), targets)
                losses += loss.item()
            clock1 = time.time()
            scheduler.step()
            print('| Epoch {:3d}, time={:5.1f}s | train_loss={:.3f} | lr={:.5f} |'.format(
                e + 1, clock1 - clock0, losses / len(trn_loader),
                self.optimizer.param_groups[0]['lr'], end=''))
        best_model = deepcopy(self.model.state_dict())
        self.model.load_state_dict(best_model)
        # EXEMPLAR MANAGEMENT -- select training subset
        self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)
        self.last_valid_out_dim = self.valid_out_dim
        if len(self._multiple_gpus) > 1:
            self.model = self.model.module

    def eval(self, t, val_loader, trained_classes, last, cur):
        """Contains the evaluation code"""
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()
            for images, true_labels, captions, targets in val_loader:
                images, targets = images.cuda(), targets.cuda()
                # images.to(self.model.feat.dtype)
                logits = self.model.forward(images)
                # logits = self.model.forward(images.to(self.model.dtype))
                cur_logits = logits[:, :trained_classes]
                loss = F.cross_entropy(logits, targets.to(self.device))

                probs = cur_logits.softmax(dim=-1).cpu()
                pred = probs.argmax(1)
                hits_tag = (pred.to(self.device) == targets.to(self.device)).float()

                cur_logits_per_image = logits[:, last:cur]
                cur_probs = cur_logits_per_image.softmax(dim=-1).cpu()
                cur_top_labels = cur_probs.argmax(1)
                cur_top_labels += last
                hits_taw = (cur_top_labels.to(self.device) == targets.to(self.device)).float()
                # Log
                total_loss += loss.data.cpu().numpy().item() * len(targets)
                total_acc_taw += hits_taw.sum().data.cpu().numpy().item()
                total_acc_tag += hits_tag.sum().data.cpu().numpy().item()
                total_num += len(targets)
            return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    def criterion(self, logits, targets, data_weights):
        loss_supervised = (self.criterion_fn(logits, targets.long()) * data_weights).mean()
        return loss_supervised

    def update_model(self, inputs, targets):

        # logits
        logits, prompt_loss = self.model(inputs, train=True)
        logits = logits[:, :self.valid_out_dim]

        # ce with heuristic
        logits[:, :self.last_valid_out_dim] = -float('inf')
        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        total_loss = self.criterion(logits, targets.long(), dw_cls)

        # ce loss
        total_loss = total_loss + prompt_loss.sum()

        # step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.detach(), logits

    def forward(self, x):
        return self.model.forward(x)[:, :self.valid_out_dim]

    def data_weighting(self):
        self.dw_k = torch.tensor(np.ones(self.valid_out_dim + 1, dtype=np.float32))
        # cuda
        if self.cuda:
            self.dw_k = self.dw_k.cuda()

    def cuda(self):
        # torch.cuda.set_device(self.device)
        self.model = self.model.cuda()
        self.criterion_fn = self.criterion_fn.cuda()
        # Multi-GPU
        if len(self._multiple_gpus) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self._multiple_gpus,
                                               output_device=self._multiple_gpus[0])
        return self

    def add_valid_output_dim(self, dim=0):
        # This function is kind of ad-hoc, but it is the simplest way to support incremental class learning
        print('Incremental class: Old valid output dimension:', self.valid_out_dim)
        self.valid_out_dim += dim
        print('Incremental class: New Valid output dimension:', self.valid_out_dim)
        return self.valid_out_dim


from torch.optim import Optimizer
import math

class _LRScheduler(object):
    def __init__(self, optimizer, last_epoch=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.
        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class CosineSchedule(_LRScheduler):

    def __init__(self, optimizer, K):
        self.K = K
        super().__init__(optimizer, -1)

    def cosine(self, base_lr):
        return base_lr * math.cos((99 * math.pi * (self.last_epoch)) / (200 * (self.K - 1)))

    def get_lr(self):
        return [self.cosine(base_lr) for base_lr in self.base_lrs]
