from torchvision import models
from utils.model.my_model import my_model

from utils.model.backbone.resnet18_no1 import resnet18_cbam1

my_models = {
    'my': my_model,
}

backbones = {
    'resnet18': models.resnet18,
    'resnet18_no1': resnet18_cbam1,
}


def prepare_model(args):
    model = my_models['my'](backbone=backbones['resnet18_no1'], pretrained=False, args=args)
    return model
