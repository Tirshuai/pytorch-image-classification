
import torch
import torch.nn as nn


class ConvBNReLU(nn.Sequential):
    """Convolution + BatchNorm + ReLU6."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        groups=1,
    ):
        padding = (kernel_size - 1) // 2

        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )


class InvertedResidual(nn.Module):
    """
    MobileNetV2 Inverted Residual Block.

    expand_ratio:
        Expansion factor for the hidden channels.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        expand_ratio,
    ):
        super().__init__()

        hidden_channels = in_channels * expand_ratio

        self.use_residual = (
            stride == 1 and in_channels == out_channels
        )

        layers = []

        # 1. Expansion: 1x1 convolution
        if expand_ratio != 1:
            layers.append(
                ConvBNReLU(
                    in_channels,
                    hidden_channels,
                    kernel_size=1,
                )
            )

        # 2. Depthwise convolution
        layers.append(
            ConvBNReLU(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                stride=stride,
                groups=hidden_channels,
            )
        )

        # 3. Projection: 1x1 convolution, without activation
        layers.extend([
            nn.Conv2d(
                hidden_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ])

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)

        if self.use_residual:
            return x + out

        return out


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10, width_mult=1.0):
        super().__init__()

        # (expand_ratio, channels, number_of_blocks, first_stride)
        settings = [
            (1, 16, 1, 1),
            (6, 24, 2, 1),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1),
        ]

        input_channels = int(32 * width_mult)

        self.stem = ConvBNReLU(
            3,
            input_channels,
            kernel_size=3,
            stride=1,
        )

        features = []
        
        for expand_ratio, channels, num_blocks, first_stride in settings:
            output_channels = int(channels * width_mult)

            for block_idx in range(num_blocks):
                stride = first_stride if block_idx == 0 else 1

                features.append(
                    InvertedResidual(
                        input_channels,
                        output_channels,
                        stride,
                        expand_ratio,
                    )
                )

                input_channels = output_channels

        self.features = nn.Sequential(*features)

        last_channels = int(1280 * width_mult)

        self.last_conv = ConvBNReLU(
            input_channels,
            last_channels,
            kernel_size=1,
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(last_channels, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.last_conv(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


def mobilenet_v2(num_classes=10, width_mult=1.0):
    return MobileNetV2(
        num_classes=num_classes,
        width_mult=width_mult,
    )
