import torch.nn as nn
import timm

def get_vit(model_name='vit_tiny_patch16_224', in_channels=1, num_classes=10, pretrained=False):

    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        in_chans=in_channels,
        num_classes=num_classes
    )
    return model