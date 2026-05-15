from .cnn_baseline import SimpleCNN
from .vision_transformer import get_vit
from .swin_transformer import get_swin
from .vit_variant import get_deit

def build_model(config):

    model_type = config.get('model_type', 'cnn').lower()
    in_channels = config.get('in_channels', 3)
    num_classes = config.get('num_classes', 10)
    pretrained = config.get('pretrained', False)
    
    if model_type == 'cnn':
        return SimpleCNN(in_channels=in_channels, num_classes=num_classes)
    
    elif model_type == 'vit':
        model_name = config.get('model_name', 'vit_tiny_patch16_224')
        return get_vit(model_name=model_name, in_channels=in_channels, num_classes=num_classes, pretrained=pretrained)
    
    elif model_type == 'swin':
        model_name = config.get('model_name', 'swin_tiny_patch4_window7_224')
        return get_swin(model_name=model_name, in_channels=in_channels, num_classes=num_classes, pretrained=pretrained)
    
    elif model_type == 'deit':
        model_name = config.get('model_name', 'deit_tiny_patch16_224')
        return get_deit(model_name=model_name, in_channels=in_channels, num_classes=num_classes, pretrained=pretrained)
    
    else:
        raise ValueError(f"Model '{model_type}' does not supported!")