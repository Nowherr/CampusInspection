import torch
import torch.nn as nn
import torchvision.models as tvm


class ResNetBackbone(nn.Module):
    """
    Wraps a torchvision ResNet so it returns 4 feature maps:
        x1: after layer1  (stride 4)
        x2: after layer2  (stride 8)
        x3: after layer3  (stride 16)
        x4: after layer4  (stride 32)

    Channel dims:
        resnet18/34 -> (64, 128, 256, 512)
        resnet50/101/152 -> (256, 512, 1024, 2048)
    """
    def __init__(self, name='resnet50', pretrained=True):
        super().__init__()
        name = name.lower()

        weights_map = {
            'resnet18':  (tvm.resnet18,  tvm.ResNet18_Weights.IMAGENET1K_V1),
            'resnet34':  (tvm.resnet34,  tvm.ResNet34_Weights.IMAGENET1K_V1),
            'resnet50':  (tvm.resnet50,  tvm.ResNet50_Weights.IMAGENET1K_V2),
            'resnet101': (tvm.resnet101, tvm.ResNet101_Weights.IMAGENET1K_V2),
            'resnet152': (tvm.resnet152, tvm.ResNet152_Weights.IMAGENET1K_V2),
        }
        if name not in weights_map:
            raise ValueError(f"Unsupported backbone: {name}")

        ctor, w = weights_map[name]
        try:
            net = ctor(weights=w if pretrained else None)
        except Exception:
            # Fallback for older torchvision
            net = ctor(pretrained=pretrained)

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

    def forward(self, x):
        x = self.stem(x)
        x1 = self.layer1(x)   # stride 4
        x2 = self.layer2(x1)  # stride 8
        x3 = self.layer3(x2)  # stride 16
        x4 = self.layer4(x3)  # stride 32
        return x1, x2, x3, x4


def get_backbone(name='resnet50', pretrained=True):
    return ResNetBackbone(name=name, pretrained=pretrained)


if __name__ == '__main__':
    bb = get_backbone('resnet18', pretrained=False)
    x = torch.randn(1, 3, 512, 512)
    outs = bb(x)
    for i, o in enumerate(outs, 1):
        print(f"x{i}: {tuple(o.shape)}")