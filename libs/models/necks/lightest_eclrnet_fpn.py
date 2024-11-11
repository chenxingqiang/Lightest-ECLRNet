import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmdet.models.builder import NECKS

class AdaptiveFeatureFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, current, upsampled):
        alpha = self.attention(current)
        return alpha * current + (1 - alpha) * upsampled

class PyramidPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.AdaptiveAvgPool2d((4, 4))
        ])
        self.conv = nn.Conv2d(in_channels * 21, out_channels, 1)

    def forward(self, x):
        features = []
        for pool in self.pools:
            features.append(pool(x))
        concat_features = torch.cat([x] + features, dim=1)
        return self.conv(concat_features)

@NECKS.register_module
class LightestECLRNetFPN(nn.Module):
    def __init__(self, in_channels, out_channels, num_outs):
        super(LightestECLRNetFPN, self).__init__()
        assert isinstance(in_channels, list)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs

        # Progressive channel reduction
        reduced_channels = [
            out_channels,
            out_channels * 3 // 4,
            out_channels // 2
        ]

        # Lateral convs with channel compression
        self.lateral_convs = nn.ModuleList()
        for i in range(len(in_channels)):
            self.lateral_convs.append(
                nn.Sequential(
                    # Depthwise separable convolution
                    nn.Conv2d(in_channels[i], in_channels[i], 3,
                             padding=1, groups=in_channels[i]),
                    nn.BatchNorm2d(in_channels[i]),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels[i], reduced_channels[i], 1),
                    nn.BatchNorm2d(reduced_channels[i])
                )
            )

        # Adaptive feature fusion
        self.fusions = nn.ModuleList([
            AdaptiveFeatureFusion(reduced_channels[i])
            for i in range(len(reduced_channels)-1)
        ])

        # Pyramid pooling modules
        self.pyramid_pools = nn.ModuleList([
            PyramidPooling(reduced_channels[i], reduced_channels[i])
            for i in range(len(reduced_channels))
        ])

    def forward(self, inputs):
        # Remove extra inputs if necessary
        if len(inputs) > len(self.in_channels):
            inputs = list(inputs[-len(self.in_channels):])

        # Build laterals with compressed channels
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # Apply pyramid pooling
        pooled = [
            pool(lateral)
            for pool, lateral in zip(self.pyramid_pools, laterals)
        ]

        # Top-down path with adaptive fusion
        for i in range(len(pooled)-1, 0, -1):
            upsampled = F.interpolate(
                pooled[i],
                size=pooled[i-1].shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            pooled[i-1] = self.fusions[i-1](pooled[i-1], upsampled)

        return tuple(pooled)