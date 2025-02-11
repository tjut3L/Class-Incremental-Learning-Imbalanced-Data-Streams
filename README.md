# Class Incremental Learning on Imbalanced Data Streams

## Environment setup:

install virtual environment:
```
pip install virtualenv
virtualenv CIR
source CIR/venv/bin/activate
pip install -r requirement.txt
```

## Data stream generation
Use this file to generate an imbalanced data stream that reflects common real-world scenarios, featuring datasets with significantly different sample sizes across various classes. This makes it especially suitable for studying the performance of algorithms under imbalance conditions. By adjusting the class ratios, you can examine the impact of varying degrees of imbalance on model performance. The generated data stream is to be used for subsequent model training and validation.
```
python ./scenario_configs/generation-car.py
python ./scenario_configs/generation-cifar100.py
python ./scenario_configs/generation-cifar10.py
python ./scenario_configs/generation-food.py
python ./scenario_configs/generation-imagenet_subset.py
python ./scenario_configs/generation-imagenet.py
```

## Training

```
  python train.py --cuda 0 --dataset CIFAR100 --config_file CIFAR100_s1.pkl
```
where the flags are explained as:
    - `--dataset`: The dataset used for training. (You can easily incorporate your own datasets to enrich your experiments.)
    - `--config_file`: Imbalanced data stream generated based on the dataset (to obtain this file for the corresponding dataset, please refer to the Data stream generation mentioned above).