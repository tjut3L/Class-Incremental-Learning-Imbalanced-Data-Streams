import argparse
import os
from collections import OrderedDict

import numpy as np
import torch
from torch import nn

import models
from models.vit_vpt import PromptedVisionTransformer
import math
import torch

from torch import nn
from typing import List, Type
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        mlp_dims: List[int],
        dropout: float = 0.1,
        nonlinearity: Type[nn.Module] = nn.ReLU,
        normalization: Type[nn.Module] = nn.BatchNorm1d,  # nn.LayerNorm,
        special_bias: bool = False,
        add_bn_first: bool = False,
    ):
        super(MLP, self).__init__()
        projection_prev_dim = input_dim
        projection_modulelist = []
        last_dim = mlp_dims[-1]
        mlp_dims = mlp_dims[:-1]

        if add_bn_first:
            if normalization is not None:
                projection_modulelist.append(normalization(projection_prev_dim))
            if dropout != 0:
                projection_modulelist.append(nn.Dropout(dropout))

        for idx, mlp_dim in enumerate(mlp_dims):
            fc_layer = nn.Linear(projection_prev_dim, mlp_dim)
            nn.init.kaiming_normal_(fc_layer.weight, a=0, mode='fan_out')
            projection_modulelist.append(fc_layer)
            projection_modulelist.append(nonlinearity())

            if normalization is not None:
                projection_modulelist.append(normalization(mlp_dim))

            if dropout != 0:
                projection_modulelist.append(nn.Dropout(dropout))
            projection_prev_dim = mlp_dim

        self.projection = nn.Sequential(*projection_modulelist)
        self.last_layer = nn.Linear(projection_prev_dim, last_dim)
        nn.init.kaiming_normal_(self.last_layer.weight, a=0, mode='fan_out')
        if special_bias:
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            torch.nn.init.constant_(self.last_layer.bias, bias_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        input_arguments:
            @x: torch.FloatTensor
        """
        x = self.projection(x)
        x = self.last_layer(x)
        return x

class ViT(nn.Module):
    """ViT-related model."""

    def __init__(self, cfg, load_pretrain=True, vis=False):
        super(ViT, self).__init__()
        parser = argparse.ArgumentParser()
        parser.add_argument('--num-tokens', default=5, type=int, help='Name of model to train')
        parser.add_argument('--location', default='prepend', type=str, help='Name of model to train')
        parser.add_argument('--vit-pool-type', default='original', type=str, help='Name of model to train')
        parser.add_argument('--initiation', default='random', type=str, help='Name of model to train')
        parser.add_argument('--project', default=-1, type=str, help='Name of model to train')
        parser.add_argument('--input-size', default=224, type=int, help='images input size')
        parser.add_argument('--pretrained', default=True, help='Load pretrained model or not')
        parser.add_argument('--deep-shared', default=False, help='Load pretrained model or not')
        parser.add_argument('--deep', default=False, help='Load pretrained model or not')
        parser.add_argument('--dropout', type=float, default=0.0, metavar='PCT', help='Dropout rate (default: 0.)')
        parser.add_argument('--num-deep-layers', type=int, default=None, help='Dropout rate (default: 0.)')
        prompt_cfg = parser.parse_args()
        self.froze_enc = False
        adapter_cfg = None
        self.build_backbone(
            prompt_cfg, cfg, adapter_cfg, load_pretrain, vis=vis)
        self.cfg = cfg
        self.setup_side()
        self.setup_head(cfg)

    def setup_side(self):
        if self.cfg.transfer_type != "side":
            self.side = None
        else:
            self.side_alpha = nn.Parameter(torch.tensor(0.0))
            m = models.alexnet(pretrained=True)
            self.side = nn.Sequential(OrderedDict([
                ("features", m.features),
                ("avgpool", m.avgpool),
            ]))
            self.side_projection = nn.Linear(9216, self.feat_dim, bias=False)

    def build_backbone(self, prompt_cfg, cfg, adapter_cfg, load_pretrain, vis):
        transfer_type = cfg.transfer_type
        self.enc, self.feat_dim = build_vit_sup_models(
            "sup_vitb16_224", 224, prompt_cfg, "/home/tjut_zhaoyishuo/clvision-challenge-23/models/", adapter_cfg, load_pretrain, vis
        )

        # linear, prompt, cls, cls+prompt, partial_1
        if transfer_type == "prompt" and prompt_cfg.location == "below":
            for k, p in self.enc.named_parameters():
                if "prompt" not in k and "embeddings.patch_embeddings.weight" not in k and "embeddings.patch_embeddings.bias" not in k:
                    p.requires_grad = False

        elif transfer_type == "prompt":
            for k, p in self.enc.named_parameters():
                if "prompt" not in k:
                    p.requires_grad = False

        elif transfer_type == "prompt+bias":
            for k, p in self.enc.named_parameters():
                if "prompt" not in k and 'bias' not in k:
                    p.requires_grad = False

        elif transfer_type == "prompt-noupdate":
            for k, p in self.enc.named_parameters():
                p.requires_grad = False

        elif transfer_type == "cls":
            for k, p in self.enc.named_parameters():
                if "cls_token" not in k:
                    p.requires_grad = False

        elif transfer_type == "cls-reinit":
            nn.init.normal_(
                self.enc.transformer.embeddings.cls_token,
                std=1e-6
            )

            for k, p in self.enc.named_parameters():
                if "cls_token" not in k:
                    p.requires_grad = False

        elif transfer_type == "cls+prompt":
            for k, p in self.enc.named_parameters():
                if "prompt" not in k and "cls_token" not in k:
                    p.requires_grad = False

        elif transfer_type == "cls-reinit+prompt":
            nn.init.normal_(
                self.enc.transformer.embeddings.cls_token,
                std=1e-6
            )
            for k, p in self.enc.named_parameters():
                if "prompt" not in k and "cls_token" not in k:
                    p.requires_grad = False

    def setup_head(self, cfg):
        self.head = MLP(
            input_dim=self.feat_dim,
            mlp_dims=[self.feat_dim] * self.cfg.mlp_num + \
                     [cfg.number_classes],  # noqa
            special_bias=True
        )

    def forward(self, x, return_feature=False):
        if self.side is not None:
            side_output = self.side(x)
            side_output = side_output.view(side_output.size(0), -1)
            side_output = self.side_projection(side_output)

        if self.froze_enc and self.enc.training:
            self.enc.eval()
        x = self.enc(x)  # batch_size x self.feat_dim

        if self.side is not None:
            alpha_squashed = torch.sigmoid(self.side_alpha)
            x = alpha_squashed * x + (1 - alpha_squashed) * side_output

        if return_feature:
            return x, x
        x = self.head(x)

        return x

    def forward_cls_layerwise(self, x):
        cls_embeds = self.enc.forward_cls_layerwise(x)
        return cls_embeds

    def get_features(self, x):
        """get a (batch_size, self.feat_dim) feature"""
        x = self.enc(x)  # batch_size x self.feat_dim
        return x


_MODEL_TYPES = {
    "vit": ViT,
}
def build_model(cfg):
    """
    build model here
    """
    # Construct the model
    train_type = "prompt"
    model = ViT(cfg)
    return model

MODEL_ZOO = {
    "swint_imagenet": "swin_tiny_patch4_window7_224.pth",
    "swint_imagenet_ssl": "moby_swin_t_300ep_pretrained.pth",
    "swins_imagenet": "swin_small_patch4_window7_224.pth",
    "swinb_imagenet_224": "swin_base_patch4_window7_224.pth",
    "swinb_imagenet_384": "swin_base_patch4_window12_384.pth",
    "swinb_imagenet22k_224":  "swin_base_patch4_window7_224_22k.pth",
    "swinb_imagenet22k_384": "swin_base_patch4_window12_384_22k.pth",
    "swinl_imagenet22k_224": "swin_large_patch4_window7_224_22k.pth",
    "sup_vitb8": "ViT-B_8.npz",
    "sup_vitb16_224": "ViT-B_16-224.npz",
    "sup_vitb16": "ViT-B_16.npz",
    "sup_vitl16_224": "ViT-L_16-224.npz",
    "sup_vitl16": "ViT-L_16.npz",
    "sup_vitb8_imagenet21k": "imagenet21k_ViT-B_8.npz",
    "sup_vitb32_imagenet21k": "imagenet21k_ViT-B_32.npz",
    "sup_vitb16_imagenet21k": "imagenet21k_ViT-B_16.npz",
    "sup_vitl16_imagenet21k": "imagenet21k_ViT-L_16.npz",
    "sup_vitl32_imagenet21k": "imagenet21k_ViT-L_32.npz",
    "sup_vith14_imagenet21k": "imagenet21k_ViT-H_14.npz",
    "mae_vith14": "mae_pretrain_vit_huge.pth",
    "mae_vitb16": "mae_pretrain_vit_base.pth",
    "mae_vitl16": "mae_pretrain_vit_large.pth",
}


def build_vit_sup_models(
        model_type, crop_size, prompt_cfg=None, model_root=None, adapter_cfg=None, load_pretrain=True, vis=False
):
    # image size is the size of actual image
    m2featdim = {
        "sup_vitb16_224": 768,
        "sup_vitb16": 768,
        "sup_vitl16_224": 1024,
        "sup_vitl16": 1024,
        "sup_vitb8_imagenet21k": 768,
        "sup_vitb16_imagenet21k": 768,
        "sup_vitb32_imagenet21k": 768,
        "sup_vitl16_imagenet21k": 1024,
        "sup_vitl32_imagenet21k": 1024,
        "sup_vith14_imagenet21k": 1280,
    }
    model = PromptedVisionTransformer(
            prompt_cfg, model_type,
            crop_size, num_classes=-1, vis=vis
        )
    if load_pretrain:
        model.load_from(np.load(os.path.join(model_root, MODEL_ZOO[model_type])))

    return model, m2featdim[model_type]