import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes = 10):
        super().__init__()
        
        self.features = nn.Sequential( #特征提取
            nn.Conv2d(
                in_channels=3, 
                out_channels=32,
                kernel_size=3,
                padding = 1
            ),
            nn.ReLU(), 
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )
        )

        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(p=0.2),
            nn.Linear(
                in_features=4096,
                out_features=num_classes
            )
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

if __name__ == "__main__":

    model = SimpleCNN(num_classes=10)

    # 模拟一个 Batch
    x = torch.randn(32, 3, 32, 32)

    output = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)

    print(model)

    total_params = sum(
        p.numel() for p in model.parameters()
    )
    print(f"Total parameters: {total_params}")

    for name, p in model.named_parameters():
        print(name, p.shape, p.numel())