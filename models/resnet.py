import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # 残差分支：第一个卷积
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 残差分支：第二个卷积
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # Shortcut 分支
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        self.layer1 = self.make_layer(in_channels=64, out_channels=64, num_blocks=2, stride=1)

        self.layer2 = self.make_layer(in_channels=64, out_channels=128, num_blocks=2, stride=2)

        self.layer3 = self.make_layer(in_channels=128, out_channels=256, num_blocks=2, stride=2)

        self.layer4 = self.make_layer(in_channels=256, out_channels=512, num_blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []

        layers.append(
            BasicBlock(in_channels, out_channels, stride)
        )

        for _ in range(1, num_blocks):
            layers.append(
                BasicBlock(out_channels, out_channels, stride=1)
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model = ResNet(num_classes=10)

    x = torch.randn(8, 3, 32, 32)
    output = model(x)

    print(output.shape)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

def resnet18(num_classes = 10):
    return ResNet(num_classes=num_classes)