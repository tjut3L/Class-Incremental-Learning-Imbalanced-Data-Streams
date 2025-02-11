#!/usr/bin/env python3
"""
Model construction functions.
"""
from tabnanny import verbose
import torch
from .vit_models import ViT
# Supported model types
_MODEL_TYPES = {
    "vit": ViT,
}


def build_model(cfg):
    """
    build model here
    """
    # Construct the model
    train_type = cfg.MODEL.TYPE
    model = _MODEL_TYPES[train_type](cfg)
    model, device = load_model_to_device(model, cfg)

    return model, device


def log_model_info(model, verbose=False):
    """Logs model info"""
    model_total_params = sum(p.numel() for p in model.parameters())
    model_grad_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)


def get_current_device():
    if torch.cuda.is_available():
        # Determine the GPU used by the current process
        cur_device = torch.cuda.current_device()
    else:
        cur_device = torch.device('cpu')
    return cur_device


def load_model_to_device(model, cfg):
    cur_device = get_current_device()
    if torch.cuda.is_available():
        # Transfer the model to the current GPU device
        model = model.cuda(device=cur_device)
        # Use multi-process data parallel model in the multi-gpu setting
        if cfg.NUM_GPUS > 1:
            # Make model replica operate on the current device
            model = torch.nn.parallel.DistributedDataParallel(
                module=model, device_ids=[cur_device], output_device=cur_device,
                find_unused_parameters=True,
            )
    else:
        model = model.to(cur_device)
    return model, cur_device
