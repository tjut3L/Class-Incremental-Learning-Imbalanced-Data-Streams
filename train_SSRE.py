#######################################
#     DO NOT CHANGE THESE IMPORTS
'''
s1 0.5
s2 0.02
s3 0.2

'''
import os
import sys
#
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from benchmarks import get_cifar_based_benchmark
from torch.optim import lr_scheduler
from utils.model import prepare_model
import argparse
import torch
from torch.nn import CrossEntropyLoss
from avalanche.training.supervised import Naive
from avalanche.training.plugins import EWCPlugin, LwFPlugin, ReplayPlugin,DKDPlugin,BiCPlugin,MIRPlugin,PASSPlugin,CoPEPlugin,SDCPlugin,SSREPlugin,IL2APlugin
from avalanche.logging import InteractiveLogger, TextLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
)


from utils.competition_plugins import (
    GPUMemoryChecker,
    RAMChecker,
    TimeChecker
)



def main(args):
    # --- Device
    device = torch.device(
        f"cuda:{args.cuda}"
        if torch.cuda.is_available() and args.cuda >= 0
        else "cpu"
    )

    # --- Benchmark
    benchmark = get_cifar_based_benchmark(scenario_config=args.config_file,
                                          seed=args.seed)
    # --- Model
    model = prepare_model(benchmark)   #SSRE
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
    competition_plugins = [
        GPUMemoryChecker(max_allowed=1000, device=device),
        RAMChecker(max_allowed=4000),
        TimeChecker(max_allowed=500)
    ]

    # --- Your Plugins
    plugins = [
        # EWCPlugin(ewc_lambda=0.1),
        # DKDPlugin(),
        # LwFPlugin(alpha=30.0),
        # MyPlugin0()
        # MIRPlugin()
        # BiCPlugin()
        # SDCPlugin()
        # PASSPlugin()
        SSREPlugin()
        # CoPEPlugin()
        # IL2APlugin()
    ]


    # --- Strategy
    cl_strategy = Naive(
        model,
        torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=5e-4),
        CrossEntropyLoss(),
        train_mb_size=16,
        train_epochs=60,  # if args.config_file == "config_s2.pkl" else 60,
        eval_mb_size=100,
        device=device,
        plugins=plugins,  # competition_plugins + plugins,
        evaluator=eval_plugin,
        # config=args.config_file,
    )
    cl_strategy.scheduler = lr_scheduler.StepLR(cl_strategy.optimizer, step_size=45, gamma=0.1)
    cl_strategy.benchmark = benchmark
    cl_strategy.pre_exp_idx = 0
    # --- Training Loops
    print(args.config_file)
    for experience in benchmark.train_stream:
        print("开始训练：experience_", experience.current_experience)
        cl_strategy.train(experience, num_workers=args.num_workers)
        if experience.current_experience>45:
            cl_strategy.eval(benchmark.test_stream)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=int, default=0,
                        help="Select zero-indexed cuda device. -1 to use CPU.")
    parser.add_argument("--config_file", type=str, default="config_s1.pkl")
    parser.add_argument("--log_file", type=str, default='./log_.txt')
    parser.add_argument("--run_name", type=str, default="test_s1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=2)

    args = parser.parse_args()

    main(args)

    # output_name = f"./pred_{args.config_file.split('.')[0]}_{args.run_name}.npy"


