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

from timm.optim import create_optimizer

#
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from benchmarks import get_cifar_based_benchmark
#
from models import L2pNet
sys.path.insert(0, "avalanche")
# from models import SlimResNet18  # FT、EWC、LWF、消融
from timm.models import create_model


import argparse
import torch
from torch.nn import CrossEntropyLoss
import torch.optim.lr_scheduler
import numpy as np

from avalanche.training.supervised import Naive
from avalanche.training.plugins import EWCPlugin, LwFPlugin, ReplayPlugin,DKDPlugin,BiCPlugin,MIRPlugin,PASSPlugin,CoPEPlugin,SDCPlugin,IL2APlugin,L2PPlugin
from avalanche.logging import InteractiveLogger, TextLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
)



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


    print(f"Creating original model: {args.model}")
    original_model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        prompt_length=args.length,
        embedding_key=args.embedding_key,
        prompt_init=args.prompt_key_init,
        prompt_pool=args.prompt_pool,
        prompt_key=args.prompt_key,
        pool_size=args.size,
        top_k=args.top_k,
        batchwise_prompt=args.batchwise_prompt,
        prompt_key_init=args.prompt_key_init,
        head_type=args.head_type,
        use_prompt_mask=args.use_prompt_mask,
    )
    original_model.to(device)
    model.to(device)

    if args.freeze:
        # all parameters are frozen for original vit model
        for p in original_model.parameters():
            p.requires_grad = False

        # freeze args.freeze[blocks, patch_embed, cls_token] parameters
        for n, p in model.named_parameters():
            if n.startswith(tuple(args.freeze)):
                p.requires_grad = False

    if args.unscale_lr:  # scaling lr by batch size
        global_batch_size = args.batch_size
    else:
        global_batch_size = args.batch_size * args.world_size
    args.lr = args.lr * global_batch_size / 256.0

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
        L2PPlugin()
    ]


    # --- Strategy
    cl_strategy = Naive(
        model,
        optimizer,
        CrossEntropyLoss().to(device),
        train_mb_size=args.batch_size,
        train_epochs=args.epochs,  # if args.config_file == "config_s2.pkl" else 60,
        eval_mb_size=100,
        device=device,
        plugins=plugins,  # competition_plugins + plugins,
        evaluator=eval_plugin,
        # config=args.config_file,
    )
    cl_strategy.args = args
    cl_strategy.benchmark = benchmark
    cl_strategy.original_model = original_model
    cl_strategy.pre_exp_idx = 0
    cl_strategy.cls_features = None
    # --- Training Loops
    print(args.config_file)
    for experience in benchmark.train_stream:
        print("开始训练：experience_", experience.current_experience)
        cl_strategy.train(experience, num_workers=args.num_workers)
        # if experience.current_experience>47:
        cl_strategy.eval(benchmark.test_stream)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=int, default=1,
                        help="Select zero-indexed cuda device. -1 to use CPU.")
    parser.add_argument("--config_file", type=str, default="config_s3.pkl")
    parser.add_argument("--log_file", type=str, default='./log_.txt')
    parser.add_argument("--run_name", type=str, default="test_s1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument('--batch_size', default=16, type=int, help='Batch size per device')
    parser.add_argument('--epochs', default=5, type=int)

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16_224', type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--pretrained', default=True, help='Load pretrained model or not')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT', help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.0, metavar='PCT', help='Drop path rate (default: 0.)')

    # Optimizer parameters
    parser.add_argument('--opt', default='adam', type=str, metavar='OPTIMIZER', help='Optimizer (default: "adam"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON', help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=(0.9, 0.999), type=float, nargs='+', metavar='BETA', help='Optimizer Betas (default: (0.9, 0.999), use opt default)')
    parser.add_argument('--clip-grad', type=float, default=1.0, metavar='NORM',  help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='weight decay (default: 0.0)')
    parser.add_argument('--reinit_optimizer', type=bool, default=True, help='reinit optimizer (default: True)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='constant', type=str, metavar='SCHEDULER', help='LR scheduler (default: "constant"')
    parser.add_argument('--lr', type=float, default=0.03, metavar='LR', help='learning rate (default: 0.03)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct', help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT', help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV', help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR', help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N', help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N', help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N', help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N', help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE', help='LR decay rate (default: 0.1)')
    parser.add_argument('--unscale_lr', type=bool, default=True, help='scaling lr by batch size (default: True)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=None, metavar='PCT', help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.0, metavar='PCT', help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel', help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1, help='Random erase count (default: 1)')

    # Data parameters
    parser.add_argument('--data-path', default='./local_datasets', type=str, help='dataset path')
    parser.add_argument('--dataset', default='Split-CIFAR100', type=str, help='dataset name')
    parser.add_argument('--shuffle', default=False, help='shuffle the data order')
    parser.add_argument('--output_dir', default='./output', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda:3', help='device to use for training / testing')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # Continual learning parameters
    parser.add_argument('--num_tasks', default=10, type=int, help='number of sequential tasks')
    parser.add_argument('--train_mask', default=True, type=bool, help='if using the class mask at training')
    parser.add_argument('--set_training_mode', default=True, type=bool, help='if using the class mask at training')
    parser.add_argument('--task_inc', default=False, type=bool, help='if doing task incremental')
    parser.add_argument('--nb_classes', default=100, type=int, help='if doing task incremental')

    # Prompt parameters
    parser.add_argument('--prompt_pool', default=True, type=bool,)
    parser.add_argument('--size', default=10, type=int,)
    parser.add_argument('--length', default=5,type=int, )
    parser.add_argument('--top_k', default=5, type=int, )
    parser.add_argument('--initializer', default='uniform', type=str,)
    parser.add_argument('--prompt_key', default=True, type=bool,)
    parser.add_argument('--prompt_key_init', default='uniform', type=str)
    parser.add_argument('--use_prompt_mask', default=False, type=bool)
    parser.add_argument('--shared_prompt_pool', default=False, type=bool)
    parser.add_argument('--shared_prompt_key', default=False, type=bool)
    parser.add_argument('--batchwise_prompt', default=True, type=bool)
    parser.add_argument('--embedding_key', default='cls', type=str)
    parser.add_argument('--predefined_key', default='', type=str)
    parser.add_argument('--pull_constraint', default=True)
    parser.add_argument('--pull_constraint_coeff', default=0.1, type=float)

    # ViT parameters
    parser.add_argument('--global_pool', default='token', choices=['token', 'avg'], type=str, help='type of global pooling for final sequence')
    parser.add_argument('--head_type', default='prompt', choices=['token', 'gap', 'prompt', 'token+prompt'], type=str, help='input type of classification head')
    parser.add_argument('--freeze', default=['blocks', 'patch_embed', 'cls_token', 'norm', 'pos_embed'], nargs='*', type=list, help='freeze part in backbone model')

    # Misc parameters
    parser.add_argument('--print_freq', type=int, default=10, help = 'The frequency of printing')
    args = parser.parse_args()

    main(args)



