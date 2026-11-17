
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 图像转换
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding = 4),
    transforms.RandomHorizontalFlip(p = 0.5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = (0.4914, 0.4822, 0.4465),
        std = (0.2470, 0.2435, 0.2616)
    )
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean = (0.4914, 0.4822, 0.4465),
        std = (0.2470, 0.2435, 0.2616)
    )
])

# 训练集
train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    transform=train_transform,
    download=False
)

# 测试集
test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    transform=test_transform,
    download=False
)

# DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

# 查看数据
print("Train size:", len(train_dataset))
print("Test size:", len(test_dataset))

image, label = train_dataset[0]

print("Image shape:", image.shape)
print("Image dtype:", image.dtype)
print("Image min:", image.min())
print("Image max:", image.max())
print("Label:", label)

for i in range(5):
    image, label = train_dataset[0]
    print(image.mean().item())

# 查看一个 Batch
images, labels = next(iter(train_loader))

print("Batch image shape:", images.shape)
print("Batch label shape:", labels.shape)
