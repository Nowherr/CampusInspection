from .backbone import *
import torch
import torch.nn as nn
import torch.nn.functional as F


class image_pooling(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(image_pooling, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        h = self.pool(x)
        h = self.conv(h)
        output = F.interpolate(h, size=x.shape[2:], mode='bilinear')
        return output


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=[6, 12, 18]):
        super(ASPP, self).__init__()
        self.dilation = dilation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.image_pool = image_pooling(self.in_channels, self.out_channels)
        self.ASPP = nn.Sequential()
        for dilate in dilation:
            self.ASPP.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, padding=dilate,
                              kernel_size=3, stride=1, dilation=dilate),
                    nn.BatchNorm2d(self.out_channels),
                    nn.ReLU(inplace=True),
                )
            )

    def forward(self, x):
        output = []
        for block in self.ASPP:
            h = block(x)
            output.append(h)
        output.append(self.image_pool(x))

        return torch.cat(output, 1)


class Deeplabv3plus(nn.Module):
    def __init__(self, backbone='resnet18', num_classes=12):
        super(Deeplabv3plus, self).__init__()
        if backbone == 'resnet18' or backbone == 'resnet34':
            self.cf_inchannels1 = 512
            self.cf_inchannels2 = 128
        else:
            self.cf_inchannels1 = 2048
            self.cf_inchannels2 = 512
        self.num_classes = num_classes
        self.backbone = get_backbone(backbone)
        self.ASPP = ASPP(in_channels=self.cf_inchannels1, out_channels=256)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=256 * 4, out_channels=self.num_classes, kernel_size=1),
            nn.BatchNorm2d(self.num_classes),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=self.num_classes+self.cf_inchannels2, out_channels=self.num_classes, kernel_size=1),
            nn.BatchNorm2d(self.num_classes),
            nn.ReLU(inplace=True)
        )

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            if isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)

    def forward(self, x):
        _, x2, _, x4 = self.backbone(x)
        h = self.ASPP(x4)
        h = self.conv1(h)
        h =F.interpolate(h, size=x2.shape[2:], mode='bilinear')
        h = self.classifier(torch.cat((h, x2), 1))
        output = F.interpolate(h, size=x.shape[2:], mode='bilinear')

        return output


if __name__ == "__main__":
    model = Deeplabv3plus(backbone='resnet18')
    model.eval()
    image = torch.randn(1, 3, 512, 512)

    print(model)
    print("input:", image.shape)
    print("output:", model(image).shape)
