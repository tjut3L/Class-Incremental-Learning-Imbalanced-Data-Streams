
from vit import VisionTransformer


def build_model(cfg):
    """
    build model here
    """
    # Construct the model
    train_type = cfg.MODEL.TYPE
    model = VisionTransformer(cfg)
    model.to(device)
    return model, device