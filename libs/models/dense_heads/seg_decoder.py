import torch
import torch.nn as nn
import torch.nn.functional as F

class LightASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3,
                              padding=6, dilation=6, groups=in_channels)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3,
                              padding=12, dilation=12, groups=in_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv4 = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        size = x.shape[2:]
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(self.pool(x))
        x4 = F.interpolate(x4, size=size, mode='bilinear', align_corners=False)
        return torch.cat([x1, x2, x3, x4], dim=1)

class ProgressiveUpsampling(nn.Module):
    def __init__(self, in_channels, num_classes, steps=3):
        super().__init__()
        self.steps = steps
        channels = in_channels
        self.convs = nn.ModuleList()

        for _ in range(steps):
            self.convs.append(nn.Sequential(
                nn.Conv2d(channels, channels // 2, 3, padding=1, groups=channels//4),
                nn.BatchNorm2d(channels // 2),
                nn.ReLU(inplace=True)
            ))
            channels //= 2

        self.final_conv = nn.Conv2d(channels, num_classes, 1)

    def forward(self, x):
        for conv in self.convs:
            x = F.interpolate(x, scale_factor=2, mode='bilinear',
                            align_corners=False)
            x = conv(x)
        return self.final_conv(x)

class FeatureWeighting(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 16, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weight = self.fc(x)
        return x * weight

class LightSegDecoder(nn.Module):
    def __init__(self, image_height, image_width, num_classes, in_channels):
        super().__init__()

        self.feature_weighting = FeatureWeighting(in_channels)
        self.aspp = LightASPP(in_channels, in_channels // 2)
        self.progressive_up = ProgressiveUpsampling(in_channels * 2, num_classes)

        target_size = (image_height // 4, image_width // 4)
        self.low_res_decoder = nn.Sequential(
            nn.Conv2d(in_channels * 2, num_classes, 1),
            nn.Upsample(size=target_size, mode='bilinear', align_corners=False)
        )

        self.training = True
        self.image_height = image_height
        self.image_width = image_width

    def forward(self, x):
        # Feature weighting
        x = self.feature_weighting(x)

        # Light ASPP
        x = self.aspp(x)

        if self.training:
            # High resolution path with progressive upsampling
            high_res = self.progressive_up(x)

            # Low resolution path for faster inference
            low_res = self.low_res_decoder(x)

            return high_res, low_res
        else:
            # Only use low resolution path during inference
            return self.low_res_decoder(x)

    def train(self, mode=True):
        self.training = mode
        super().train(mode)
        return self