from .backbone import *
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImagePooling(nn.Module):
    """Global image pooling branch in ASPP."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = x.shape[2:]
        h = self.pool(x)
        h = self.conv(h)
        return F.interpolate(h, size=size, mode='bilinear', align_corners=False)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3,
                      padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ASPP(nn.Module):
    """
    Standard DeepLabV3+ ASPP:
      - 1x1 conv branch
      - 3 dilated conv branches
      - global image pooling branch
      - concat (5 * out_channels) -> 1x1 project -> out_channels
    """
    def __init__(self, in_channels, out_channels=256, dilations=(6, 12, 18)):
        super().__init__()
        branches = []
        # 1x1 branch
        branches.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ))
        # dilated branches
        for d in dilations:
            branches.append(ASPPConv(in_channels, out_channels, d))
        # image pooling branch
        branches.append(ImagePooling(in_channels, out_channels))

        self.branches = nn.ModuleList(branches)

        # project back to out_channels (this is the key fix!)
        self.project = nn.Sequential(
            nn.Conv2d(len(branches) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        feats = [b(x) for b in self.branches]
        return self.project(torch.cat(feats, dim=1))


class Deeplabv3plus(nn.Module):
    """
    DeepLabV3+ with proper decoder:
      high-level ---ASPP---> 256ch --upsample--+
                                                |-- concat --> 3x3 --> 3x3 --> 1x1 classifier
      low-level ---1x1 conv---> 48ch -----------+
    """
    def __init__(self, backbone='resnet50', num_classes=12, low_level_channels=48):
        super().__init__()
        if backbone in ('resnet18', 'resnet34'):
            high_level_in = 512
            low_level_in = 128
        else:
            high_level_in = 2048
            low_level_in = 512

        self.num_classes = num_classes
        self.backbone = get_backbone(backbone)

        # ASPP: high-level feature -> 256 channels
        self.aspp = ASPP(high_level_in, out_channels=256)

        # Low-level projection: reduce channels to 48 (not num_classes!)
        self.low_level_proj = nn.Sequential(
            nn.Conv2d(low_level_in, low_level_channels, 1, bias=False),
            nn.BatchNorm2d(low_level_channels),
            nn.ReLU(inplace=True),
        )

        # Decoder: concat(256 + 48) -> 3x3 -> 3x3 -> classifier
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + low_level_channels, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.classifier = nn.Conv2d(256, num_classes, 1)

        self._init_weights()

    def _init_weights(self):
        # Initialize only new layers (do NOT override pretrained backbone weights)
        for name, m in self.named_modules():
            if name.startswith('backbone'):
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        input_size = x.shape[2:]
        # backbone returns 4 feature maps; we use x2 (low-level) and x4 (high-level)
        _, low_level, _, high_level = self.backbone(x)

        # high-level branch
        h = self.aspp(high_level)
        h = F.interpolate(h, size=low_level.shape[2:],
                          mode='bilinear', align_corners=False)

        # low-level branch
        low = self.low_level_proj(low_level)

        # decode
        h = torch.cat([h, low], dim=1)
        h = self.decoder(h)
        h = self.classifier(h)

        return F.interpolate(h, size=input_size, mode='bilinear', align_corners=False)


if __name__ == "__main__":
    model = Deeplabv3plus(backbone='resnet50', num_classes=12)
    model.eval()
    image = torch.randn(1, 3, 512, 512)
    print("input:", image.shape)
    print("output:", model(image).shape)
    print(f"#params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")