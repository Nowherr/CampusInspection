from .SegNet import SegNet


def get_model(num_classes=12, pretrained=True, **kwargs):
    """
    Factory for SegNet.

    Args:
        num_classes: number of output classes.
        pretrained: if True, load VGG16 ImageNet pretrained weights into encoder.
        **kwargs:   reserved for future models.
    """
    return SegNet(num_classes=num_classes, pretrained=pretrained, **kwargs)