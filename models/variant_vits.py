from timm.data import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    IMAGENET_INCEPTION_MEAN,
    IMAGENET_INCEPTION_STD,
)
from timm.models.helpers import build_model_with_cfg, resolve_pretrained_cfg
from timm.models.nest import resize_pos_embed


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "fixed_input_size": True,
        "mean": IMAGENET_INCEPTION_MEAN,
        "std": IMAGENET_INCEPTION_STD,
        "first_conv": "patch_embed.proj",
        **kwargs,
    }


from timm.models import vision_transformer

model_cfgs = {
    "vit_base_patch16_224":
        dict(patch_size=16, embed_dim=768, depth=12, num_heads=12),
    "vit_base_patch32_224":
        dict(patch_size=32, embed_dim=768, depth=12, num_heads=12),

    # dino
    "vit_base_patch16_224_dino":
        dict(patch_size=16, embed_dim=768, depth=12, num_heads=12),
    "vit_base_patch8_224_dino":
        dict(patch_size=8, embed_dim=768, depth=12, num_heads=12),
    "vit_small_patch16_224_dino":
        dict(patch_size=16, embed_dim=384, depth=12, num_heads=6),
    "vit_small_patch8_224_dino":
        dict(patch_size=8, embed_dim=384, depth=12, num_heads=6),

    "vit_base_patch16_clip_224_openai_fp16": {
        "model_config": dict(input_resolution=224, patch_size=16, width=768, layers=12, heads=12, output_dim=512),
        "weights_path": '/data/clip_models/vision/vit_base_patch16_224_clip_fp16.pth'},
    "vit_base_patch32_clip_224_openai_fp16": {
        "model_config": dict(input_resolution=224, patch_size=32, width=768, layers=12, heads=12, output_dim=512),
        "weights_path": '/data/clip_models/vision/vit_base_patch32_224_clip_fp16.pth'},
    "vit_base_patch16_clip_224_openai_fp32": {
        "model_config": dict(input_resolution=224, patch_size=16, width=768, layers=12, heads=12, output_dim=512),
        "weights_path": '/data/clip_models/vision/vit_base_patch16_224_clip_fp32.pth'},
    "vit_base_patch32_clip_224_openai_fp32": {
        "model_config": dict(input_resolution=224, patch_size=32, width=768, layers=12, heads=12, output_dim=512),
        "weights_path": '/data/clip_models/vision/vit_base_patch32_224_clip_fp32.pth'},

    # mae
    "vit_base_patch16_224_mae":
        dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, num_classes=0),

    # deit
    "deit_tiny_patch16_224":
        dict(patch_size=16, embed_dim=192, depth=12, num_heads=3),
    "deit_small_patch16_224":
        dict(patch_size=16, embed_dim=384, depth=12, num_heads=6),
    "deit_base_patch16_224":
        dict(patch_size=16, embed_dim=768, depth=12, num_heads=12),

    "deit_tiny_distilled_patch16_224":
        dict(patch_size=16, embed_dim=192, depth=12, num_heads=3),
    "deit_small_distilled_patch16_224":
        dict(patch_size=16, embed_dim=384, depth=12, num_heads=6),
    "deit_base_distilled_patch16_224":
        dict(patch_size=16, embed_dim=768, depth=12, num_heads=12),
}

clip_cfgs = {
    "vit_base_patch16_clip_224_openai_fp16": "ViT-B/16",
    "vit_base_patch32_clip_224_openai_fp16": "ViT-B/32"
}

default_cfgs = {
    # patch models (weights from official Google JAX impl)

    "vit_base_patch16_224_augreg2_in21k_ft_in1k": _cfg(
        hf_hub_id='timm/vit_base_patch16_224.augreg2_in21k_ft_in1k'
    ),
    "vit_tiny_patch16_224": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/"
            "Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz"
    ),
    "vit_tiny_patch16_384": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/"
            "Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz",
        input_size=(3, 384, 384),
        crop_pct=1.0,
    ),
    "vit_small_patch32_224": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/"
            "S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz"
    ),
    "vit_small_patch32_384": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/"
            "S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz",
        input_size=(3, 384, 384),
        crop_pct=1.0,
    ),
    "vit_small_patch16_224": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/"
            "S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz"
    ),
    "vit_small_patch16_384": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/"
            "S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz",
        input_size=(3, 384, 384),
        crop_pct=1.0,
    ),
    "vit_base_patch32_224": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/"
            "B_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz"
    ),
    "vit_base_patch32_384": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/"
            "B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz",
        input_size=(3, 384, 384),
        crop_pct=1.0,
    ),
    "vit_base_patch16_224": _cfg(
        # url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth"
        # url = "https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz"  #SLCA
        # url="https://storage.googleapis.com/vit_models/augreg/B_16-i1k-300ep-lr_0.001-aug_strong2-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz"
        # url="https://storage.googleapis.com/vit_models/augreg/"
        #     "B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz"
        url="https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz"  # L2P DualPrompt
    ),
    "vit_base_patch16_384": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/"
            "B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz",
        input_size=(3, 384, 384),
        crop_pct=1.0,
    ),
    "vit_base_patch8_224": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/"
            "B_8-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz"
    ),
    "vit_large_patch32_224": _cfg(
        url="",  # no official model weights for this combo, only for in21k
    ),
    "vit_large_patch32_384": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth",
        input_size=(3, 384, 384),
        crop_pct=1.0,
    ),
    "vit_large_patch16_224": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/"
            "L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz"
    ),
    "vit_large_patch16_384": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/"
            "L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.npz",
        input_size=(3, 384, 384),
        crop_pct=1.0,
    ),
    "vit_huge_patch14_224": _cfg(url=""),
    "vit_giant_patch14_224": _cfg(url=""),
    "vit_gigantic_patch14_224": _cfg(url=""),
    # patch models, imagenet21k (weights from official Google JAX impl)
    "vit_tiny_patch16_224_in21k": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz",
    ),
    "vit_small_patch32_224_in21k": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz",
    ),
    "vit_small_patch16_224_in21k": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz",
    ),
    "vit_base_patch32_224_in21k": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/B_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.03-do_0.0-sd_0.0.npz",
    ),
    "vit_base_patch16_224_in21k": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz",
    ),
    "vit_base_patch8_224_in21k": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/B_8-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz",
    ),
    "vit_large_patch32_224_in21k": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth",
    ),
    "vit_large_patch16_224_in21k": _cfg(
        url="https://storage.googleapis.com/vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1.npz",
    ),
    "vit_huge_patch14_224_in21k": _cfg(
        url="https://storage.googleapis.com/vit_models/imagenet21k/ViT-H_14.npz",
        hf_hub="timm/vit_huge_patch14_224_in21k",
    ),
    # SAM trained models (https://arxiv.org/abs/2106.01548)
    "vit_base_patch32_sam_224": _cfg(
        url="https://storage.googleapis.com/vit_models/sam/ViT-B_32.npz"
    ),
    "vit_base_patch16_sam_224": _cfg(
        url="https://storage.googleapis.com/vit_models/sam/ViT-B_16.npz"
    ),
    # deit models (FB weights)
    "deit_tiny_patch16_224": _cfg(
        # url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        file='/home/jskj_taozhe/.cache/torch/hub/checkpoints/deit_tiny_patch16_224-a1311bcf.pth'
    ),
    "deit_small_patch16_224": _cfg(
        # url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        file='/home/jskj_taozhe/.cache/torch/hub/checkpoints/deit_small_patch16_224-cd65a155.pth'
    ),
    "deit_base_patch16_224": _cfg(
        # url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        file='/home/jskj_taozhe/.cache/torch/hub/checkpoints/deit_base_patch16_224-b5f2ef4d.pth'
    ),
    "deit_base_patch16_384": _cfg(
        url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        input_size=(3, 384, 384),
        crop_pct=1.0,
    ),
    "deit_tiny_distilled_patch16_224": _cfg(
        url="https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    ),
    "deit_small_distilled_patch16_224": _cfg(
        url="https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    ),
    "deit_base_distilled_patch16_224": _cfg(
        url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    ),
    "deit_base_distilled_patch16_384": _cfg(
        url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        input_size=(3, 384, 384),
        crop_pct=1.0,
    ),
    # ViT ImageNet-21K-P pretraining by MILL
    "vit_base_patch16_224_miil_in21k": _cfg(
        url="https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/timm/vit_base_patch16_224_in21k_miil.pth",
        mean=(0, 0, 0),
        std=(1, 1, 1),
        crop_pct=0.875,
        interpolation="bilinear",
    ),
    "vit_base_patch16_224_miil": _cfg(
        url="https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/timm"
            "/vit_base_patch16_224_1k_miil_84_4.pth",
        mean=(0, 0, 0),
        std=(1, 1, 1),
        crop_pct=0.875,
        interpolation="bilinear",
    ),
    # patch models, paper (weights from official Google JAX impl)
    "vit_b32": _cfg(
        url="https://storage.googleapis.com/vit_models/imagenet21k%2Bimagenet2012/ViT-B_32.npz",
    ),
    "vit_b16": _cfg(
        url="https://storage.googleapis.com/vit_models/imagenet21k%2Bimagenet2012/ViT-B_16.npz",
    ),
    "vit_b8": _cfg(
        url="https://storage.googleapis.com/vit_models/imagenet21k%2Bimagenet2012/ViT-B_8.npz",
    ),
    "vit_l32": _cfg(
        url="https://storage.googleapis.com/vit_models/imagenet21k%2Bimagenet2012/ViT-L_32.npz",
    ),
    "vit_l16": _cfg(
        url="https://storage.googleapis.com/vit_models/imagenet21k%2Bimagenet2012/ViT-L_16.npz",
    ),
    # patch models, paper imagenet21k (weights from official Google JAX impl)
    "vit_b32_in21k": _cfg(
        url="https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_32.npz",
    ),
    "vit_b16_in21k": _cfg(
        url="https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz",
    ),
    "vit_b8_in21k": _cfg(
        url="https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_8.npz",
    ),
    "vit_l32_in21k": _cfg(
        url="https://storage.googleapis.com/vit_models/imagenet21k/ViT-L_32.npz",
    ),
    "vit_l16_in21k": _cfg(
        url="https://storage.googleapis.com/vit_models/imagenet21k/ViT-L_16.npz",
    ),

    # DINO pretrained - https://arxiv.org/abs/2104.14294 (no classifier head, for fine-tune only)
    'vit_small_patch16_224_dino': _cfg(
        url='https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),
    'vit_small_patch8_224_dino': _cfg(
        url='https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),
    'vit_base_patch16_224_dino': _cfg(
        url='https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),
    'vit_base_patch8_224_dino': _cfg(
        url='https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),
}
def checkpoint_filter_fn(state_dict, model, adapt_layer_scale=False):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    import re
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']

    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape[1] != model.pos_embed.shape[1]:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(
                v,
                model.pos_embed,
                0 if getattr(model, 'no_embed_class') else getattr(model, 'num_prefix_tokens', 1),
                model.patch_embed.grid_size
            )
        elif adapt_layer_scale and 'gamma_' in k:
            # remap layer-scale gamma into sub-module (deit3 models)
            k = re.sub(r'gamma_([0-9])', r'ls\1.gamma', k)
        elif 'pre_logits' in k:
            # NOTE representation layer removed as not used in latest 21k/1k pretrained weights
            continue
        out_dict[k] = v
    return out_dict

def _create_vision_transformer(VisionTransformer, variant, pretrained=False, pretrained_cfg=None, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')
    kwargs.pop('pretrained_cfg', None)

    pretrained_cfg = resolve_pretrained_cfg(variant, pretrained_cfg=kwargs.pop('pretrained_cfg', None))

    # pretrained_cfg = pretrained_cfg or default_cfgs[variant]
    # pretrained_cfg['custom_load'] = 'npz' in pretrained_cfg['url']
    # print(pretrained_cfg)
    model = build_model_with_cfg(
        VisionTransformer, variant, pretrained,
        pretrained_cfg=pretrained_cfg,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load='npz' in pretrained_cfg['url'],
        **kwargs
    )
    return model
