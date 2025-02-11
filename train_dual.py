#######################################
#     DO NOT CHANGE THESE IMPORTS
'''
s1 0.5
s2 0.02
s3 0.2

'''
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import sys
import random
from timm.optim import create_optimizer
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from benchmarks import get_cifar_based_benchmark
from timm.models import create_model
from models.DualNet import VisionTransformer
#####################################
# from models import SlimResNet18
import argparse
import torch
from torch.nn import CrossEntropyLoss
import torch.optim.lr_scheduler
import numpy as np
from avalanche.training.supervised import Naive
from avalanche.training.plugins import EWCPlugin, LwFPlugin, ReplayPlugin,DKDPlugin,BiCPlugin,MIRPlugin,PASSPlugin,CoPEPlugin,SDCPlugin,IL2APlugin,L2PPlugin,DualPlugin,DualPlugin_Noaug
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
        if torch.cuda.is_available()
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
        use_g_prompt=args.use_g_prompt,
        g_prompt_length=args.g_prompt_length,
        g_prompt_layer_idx=args.g_prompt_layer_idx,
        use_prefix_tune_for_g_prompt=args.use_prefix_tune_for_g_prompt,
        use_e_prompt=args.use_e_prompt,
        e_prompt_layer_idx=args.e_prompt_layer_idx,
        use_prefix_tune_for_e_prompt=args.use_prefix_tune_for_e_prompt,
        same_key_value=args.same_key_value,
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
        # LwFPlugin(alpha=10.0),
        DualPlugin()
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
    # cl_strategy.eval(benchmark.test_stream)
    for experience in benchmark.train_stream:
        print("开始训练：experience_", experience.current_experience)
        cl_strategy.train(experience, num_workers=args.num_workers)
        # if experience.current_experience>47:
        cl_strategy.eval(benchmark.test_stream)



if __name__ == "__main__":
    parser = argparse.ArgumentParser('DualPrompt training and evaluation configs')

    config = parser.parse_known_args()[-1][0]

    subparser = parser.add_subparsers(dest='subparser_name')

    if config == 'cifar100_dualprompt':
        from configs.cifar100_dualprompt import get_args_parser

        config_parser = subparser.add_parser('cifar100_dualprompt', help='Split-CIFAR100 DualPrompt configs')
    get_args_parser(config_parser)

    args = parser.parse_args()
    main(args)
    args = parser.parse_args()
    main(args)



