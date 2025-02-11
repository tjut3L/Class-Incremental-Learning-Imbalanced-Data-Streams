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
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from benchmarks import get_cifar_based_benchmark
from timm.models import create_model
#####################################
# from models import SlimResNet18
import argparse
import torch
from torch.nn import CrossEntropyLoss
import torch.optim.lr_scheduler
import numpy as np
from avalanche.training.supervised import Naive
from avalanche.training.plugins import EWCPlugin, LwFPlugin, ReplayPlugin\
    ,DKDPlugin,BiCPlugin,MIRPlugin,PASSPlugin,CoPEPlugin,SDCPlugin,IL2APlugin,L2PPlugin,DualPlugin,CodaPlugin
from avalanche.logging import InteractiveLogger, TextLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
)
from models.DualNet import _create_vision_transformer
from models.CODANet import CODANet

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
    prompt_para = [50,[100,8,0.0]]
    print(f"Creating model: {args.model}")
    model = CODANet(
        args.model,100,True,prompt_para
    )
    model.to(device)

    for name, parameter in model.named_parameters():
        if 'prompt' not in name and 'head' not in name:
            parameter.requires_grad = False
    params = list(model.prompt.parameters()) + \
             list(model.head.parameters())
    # double check, 检查需要更新的参数列表
    enabled = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            enabled.add(name)
    print(enabled)
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.WD, betas=(args.momentum, 0.999))


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
        CodaPlugin()
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
    cl_strategy.pre_exp_idx = 0
    cl_strategy.cls_features = None
    # --- Training Loops
    print(args.config_file)
    for experience in benchmark.train_stream:
        print("开始训练：experience_", experience.current_experience)
        cl_strategy.train(experience, num_workers=args.num_workers)
        if experience.current_experience>47:
            cl_strategy.eval(benchmark.test_stream)


    def _get_optimizer():
        """Returns the optimizer"""



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=int, default=4,
                        help="Select zero-indexed cuda device. -1 to use CPU.")
    parser.add_argument("--config_file", type=str, default="config_s3.pkl")
    parser.add_argument("--log_file", type=str, default='./log_CODA3.txt')
    parser.add_argument("--run_name", type=str, default="test_s1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--nb_classes", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument('--model', default='vit_base_patch16_224', type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--log_dir', type=str, default="outputs/cifar-100/10-task/coda-p",
                         help="Save experiments results in dir for future plotting!")
    parser.add_argument('--learner_type', type=str, default='prompt', help="The type (filename) of learner")
    parser.add_argument('--learner_name', type=str, default='CODAPrompt', help="The class name of learner")
    parser.add_argument('--debug_mode', type=int, default=0, metavar='N',
                        help="activate learner specific settings for debug_mode")
    parser.add_argument('--repeat', type=int, default=1, help="Repeat the experiment N times")
    parser.add_argument('--overwrite', type=int, default=0, metavar='N', help='Train regardless of whether saved model exists')

    # CL Args
    parser.add_argument('--oracle_flag', default=False, action='store_true', help='Upper bound for oracle')
    parser.add_argument('--upper_bound_flag', default=False, action='store_true', help='Upper bound')
    parser.add_argument('--memory', type=int, default=0, help="size of memory for replay")
    parser.add_argument('--temp', type=float, default=2., dest='temp', help="temperature for distillation")
    parser.add_argument('--DW', default=False, action='store_true', help='dataset balancing')
    parser.add_argument('--prompt_param', nargs="+", type=float, default=[100, 8, 0.0],
                         help="e prompt pool size, e prompt length, g prompt length")
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--WD', type=float, default=0.0, help='weight decay (default: 0.0)')
    # Config Arg
    parser.add_argument('--config', type=str, default="configs/cifar-100_prompt.yaml",
                         help="yaml experiment config input")
    args = parser.parse_args()

    main(args)




