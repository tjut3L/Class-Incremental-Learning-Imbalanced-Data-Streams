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

from avalanche.benchmarks import classification_subset
from benchmarks import get_cifar_based_benchmark
from utils.model import my_model

# from models import SlimResNet18  # Our
sys.path.insert(0, "avalanche")
from models import SlimResNet18  # FT、EWC、LWF、消融
from utils.model import prepare_model


#######################################
# from models import SlimResNet18
import argparse
import torch
from torch.nn import CrossEntropyLoss
import torch.optim.lr_scheduler
from torch.utils.data import DataLoader
import numpy as np

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



from strategies.my_strategy import MyStrategy

from strategies.my_plugin_6 import MyPlugin6
from strategies.my_plugin_5 import MyPlugin5
from strategies.my_plugin_ablaion import MyPlugin_Ablation
from strategies.my_plugin_4 import MyPlugin4
from strategies.my_plugin_3 import MyPlugin3
from strategies.my_plugin_2 import MyPlugin2
from strategies.my_plugin_1 import MyPlugin1
from strategies.my_plugin_wo_proto import MyPlugin_Wo_Proto
from strategies.my_plugin_wo_AAnet import MyPlugin_Wo_AAnet
from strategies.my_plugin_wo_Lwf import MyPlugin_Wo_Lwf
from strategies.my_plugin_0 import MyPlugin0
from strategies.my_plugin_0_DA import MyPluginDA

from strategies.myplugin_tiny import MyPluginTiny


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
    # model = prepare_model(benchmark)   #SSRE
    model = SlimResNet18(benchmark.n_classes)
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
        LwFPlugin(alpha=30.0),
        # MyPlugin0()
        # MIRPlugin()
        # BiCPlugin()
        # SDCPlugin()
        # PASSPlugin()
        # SSREPlugin()
        # CoPEPlugin()
        # IL2APlugin()
    ]


    # --- Strategy
    cl_strategy = Naive(
        model,
        torch.optim.Adam(model.parameters(), lr=0.001),
        CrossEntropyLoss(),
        train_mb_size=16,
        train_epochs=1,  # if args.config_file == "config_s2.pkl" else 60,
        eval_mb_size=100,
        device=device,
        plugins=plugins,  # competition_plugins + plugins,
        evaluator=eval_plugin,
        # config=args.config_file,
    )
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
    parser.add_argument("--cuda", type=int, default=2,
                        help="Select zero-indexed cuda device. -1 to use CPU.")
    parser.add_argument("--config_file", type=str, default="imagenet_sub_3.pkl")
    parser.add_argument("--log_file", type=str, default='./log_.txt')
    parser.add_argument("--run_name", type=str, default="test_s1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=2)

    args = parser.parse_args()

    main(args)

    # output_name = f"./pred_{args.config_file.split('.')[0]}_{args.run_name}.npy"


