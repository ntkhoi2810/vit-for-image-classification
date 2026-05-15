import timm

def get_deit(model_name='deit_tiny_patch16_224', in_channels=1, num_classes=10, pretrained=False):

    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        in_chans=in_channels,
        num_classes=num_classes
    )
    return model