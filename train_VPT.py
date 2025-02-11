#######################################
#     DO NOT CHANGE THESE IMPORTS
'''
s1 0.5
s2 0.02
s3 0.2

'''
import os
import sys
import random
from time import sleep
import math
from torch.optim.lr_scheduler import LambdaLR
from iopath import PathManager
from timm.optim import create_optimizer

#
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from benchmarks import get_cifar_based_benchmark


#######################################
# from models import SlimResNet18
import argparse
import torch
from torch.nn import CrossEntropyLoss
import torch.optim.lr_scheduler
from torch.utils.data import DataLoader
import numpy as np

from avalanche.training.supervised import Naive
from avalanche.training.plugins import EWCPlugin, LwFPlugin, ReplayPlugin,DKDPlugin,BiCPlugin,MIRPlugin,PASSPlugin,CoPEPlugin,SDCPlugin,IL2APlugin,L2PPlugin,VPTPlugin
from avalanche.logging import InteractiveLogger, TextLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
)
from configs.config import get_cfg
from models.VPTmodel import build_model

def main(args):
    # --- Device
    device = torch.device(
        f"cuda:{args.cuda}"
        if torch.cuda.is_available() and args.cuda >= 0
        else "cpu"
    )

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print(f"Creating model: {args.model}")
    model = build_model(args)
    model.to(device)
    #
    # if args.freeze:
    #     # all parameters are frozen for original vit model
    #     # freeze args.freeze[blocks, patch_embed, cls_token] parameters
    #     for n, p in model.named_parameters():
    #         if n.startswith(tuple(args.freeze)):
    #             p.requires_grad = False
    #
    # if args.unscale_lr:  # scaling lr by batch size
    #     global_batch_size = args.batch_size
    # else:
    #     global_batch_size = args.batch_size * args.world_size
    args.lr = args.lr

    optimizer = create_optimizer(args, model)


    # --- Benchmark
    benchmark = get_cifar_based_benchmark(scenario_config=args.config_file,
                                          seed=args.seed)

    # --- Logger and metrics
    interactive_logger = InteractiveLogger()
    text_logger = TextLogger(open(args.log_file, 'w'))
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(
            minibatch=True, epoch=True, experience=True, stream=True
        ),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        loggers=[interactive_logger, text_logger],
    )

    # --- Competition Plugins
    # DO NOT REMOVE OR CHANGE THESE PLUGINS:
    # --- Your Plugins
    plugins = [
        VPTPlugin()
    ]
    scheduler = make_scheduler(optimizer,args)

    # --- Strategy
    cl_strategy = Naive(
        model,
        optimizer,
        CrossEntropyLoss().to(device),
        train_mb_size=args.batch_size,
        train_epochs= args.epochs,  # if args.config_file == "config_s2.pkl" else 60,
        eval_mb_size=100,
        device=device,
        plugins=plugins,  # competition_plugins + plugins,
        evaluator=eval_plugin,
        # config=args.config_file,
    )
    cl_strategy.args = args
    cl_strategy.scheduler = scheduler
    cl_strategy.benchmark = benchmark
    cl_strategy.pre_exp_idx = 0
    cl_strategy.cls_features = None
    # --- Training Loops
    print(args.config_file)
    for experience in benchmark.train_stream:
        print("开始训练：experience_", experience.current_experience)
        cl_strategy.train(experience, num_workers=args.num_workers)
        if experience.current_experience>47:

            cl_strategy.eval(benchmark.test_stream)
import torch.optim as optim

def make_scheduler(
    optimizer, train_params
):
    warmup = train_params.warmup_epoch
    total_iters = train_params.total_epoch

    scheduler = WarmupCosineSchedule(
            optimizer,
            warmup_steps=warmup,
            t_total=total_iters
        )
    return scheduler

class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps`.
        Decreases learning rate from 1. to 0. over remaining
            `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate
            follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(
            1, self.t_total - self.warmup_steps))
        return max(
            0.0,
            0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress))
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=int, default=2,
                        help="Select zero-indexed cuda device. -1 to use CPU.")
    parser.add_argument("--config_file", type=str, default="config_s3.pkl")
    parser.add_argument("--log_file", type=str, default='./log_.txt')
    parser.add_argument("--run_name", type=str, default="test_s1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--mlp_num", type=int, default=1)
    parser.add_argument("--number_classes", type=int, default=100)

    parser.add_argument('--batch_size', default=16, type=int, help='Batch size per device')
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--model', default='vit_base_patch16_224', type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--transfer-type', default='prompt', type=str, metavar='MODEL', help='Name of model to train')
    #opti para
    parser.add_argument('--opt', default='adam', type=str, metavar='OPTIMIZER', help='Optimizer (default: "adam"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=(0.9, 0.999), type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: (0.9, 0.999), use opt default)')
    parser.add_argument('--clip-grad', type=float, default=1.0, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='weight decay (default: 0.0)')
    parser.add_argument('--reinit_optimizer', type=bool, default=True, help='reinit optimizer (default: True)')
    #scheduler para
    parser.add_argument('--warmup_epoch', default=5, type=int, help='Batch size per device')
    parser.add_argument('--total_epoch', default=10, type=int, help='Batch size per device')


    parser.add_argument(
        "--config-file1", default="./configs/config.py", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--train-type", default="prompt", help="training types")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    main(args)



