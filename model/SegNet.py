import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def crop(upsampled, bypass):
    h1, w1 = upsampled.shape[2], upsampled.shape[3]
    h2, w2 = bypass.shape[2], bypass.shape[3]

    deltah = h2 - h1
    deltaw = w2 - w1

    pad_top = deltah // 2
    pad_bottom = deltah - pad_top
    pad_left = deltaw // 2
    pad_right = deltaw - pad_left

    upsampled_padded = F.pad(upsampled, (pad_left, pad_right, pad_top, pad_bottom), "constant", 0)
    return upsampled_padded


class SegNet(nn.Module):
    def __init__(self, num_classes=12, pretrained=True):
        """
        Args:
            num_classes: number of output classes
            pretrained: if True, load VGG16 ImageNet pretrained weights into encoder
        """
        super(SegNet, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.encoder4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.encoder5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.decoder1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.decoder4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.decoder5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

        self.max_pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.max_uppool = nn.MaxUnpool2d(2, 2)

        # Step 1: Kaiming init for all layers (fallback for decoder and final layer)
        self.initialize_weights()

        # Step 2: Override encoder weights with VGG16 pretrained weights
        if pretrained:
            self.load_vgg16_pretrained()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def load_vgg16_pretrained(self):
        """Load VGG16 ImageNet pretrained weights into the 13 encoder conv layers."""
        try:
            vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        except Exception:
            # Compatibility with older torchvision
            vgg16 = models.vgg16(pretrained=True)

        vgg_conv_layers = [m for m in vgg16.features.modules() if isinstance(m, nn.Conv2d)]

        encoders = [self.encoder1, self.encoder2, self.encoder3, self.encoder4, self.encoder5]
        segnet_conv_layers = []
        for enc in encoders:
            for layer in enc:
                if isinstance(layer, nn.Conv2d):
                    segnet_conv_layers.append(layer)

        assert len(segnet_conv_layers) == 13, f"Expected 13 encoder conv layers, got {len(segnet_conv_layers)}"
        assert len(vgg_conv_layers) >= 13, f"VGG16 should have >=13 conv layers, got {len(vgg_conv_layers)}"

        with torch.no_grad():
            for seg_conv, vgg_conv in zip(segnet_conv_layers, vgg_conv_layers[:13]):
                seg_conv.weight.data.copy_(vgg_conv.weight.data)
                if seg_conv.bias is not None and vgg_conv.bias is not None:
                    seg_conv.bias.data.copy_(vgg_conv.bias.data)

        print("✅ Loaded VGG16 ImageNet pretrained weights into SegNet encoder (13 conv layers).")

    def forward(self, x):
        x1 = self.encoder1(x)
        x, pool_indices1 = self.max_pool(x1)
        x2 = self.encoder2(x)
        x, pool_indices2 = self.max_pool(x2)
        x3 = self.encoder3(x)
        x, pool_indices3 = self.max_pool(x3)
        x4 = self.encoder4(x)
        x, pool_indices4 = self.max_pool(x4)
        x5 = self.encoder5(x)
        x, pool_indices5 = self.max_pool(x5)

        x = self.max_uppool(x, pool_indices5)
        x = crop(x, x5)
        x = self.decoder1(x)
        x = self.max_uppool(x, pool_indices4)
        x = crop(x, x4)
        x = self.decoder2(x)
        x = self.max_uppool(x, pool_indices3)
        x = crop(x, x3)
        x = self.decoder3(x)
        x = self.max_uppool(x, pool_indices2)
        x = crop(x, x2)
        x = self.decoder4(x)
        x = self.max_uppool(x, pool_indices1)
        x = crop(x, x1)
        x = self.decoder5(x)

        return x


if __name__ == '__main__':
    model = SegNet(num_classes=12, pretrained=True)
    img = torch.randn(1, 3, 224, 224)
    output = model(img)
    print(output.size())