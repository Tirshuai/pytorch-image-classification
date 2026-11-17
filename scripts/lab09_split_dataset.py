import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

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

train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    transform=train_transform,
    download=False
)

val_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    transform=test_transform,
    download=False
)

generator = torch.Generator().manual_seed(42)

indices = torch.randperm(
    len(train_dataset),
    generator = generator
).tolist()

train_indices = indices[:45000]
val_indices = indices[45000:]

train_subset = Subset(train_dataset, train_indices)
val_subset = Subset(val_dataset, val_indices)

# print(f"Train samples: {len(train_subset)}")
# print(f"Validation samples: {len(val_subset)}")

# print("Train image shape:", train_subset[0][0].shape)
# print("Validation image shape:", val_subset[0][0].shape)

# print("Train image range:",
#       train_subset[0][0].min().item(),
#       train_subset[0][0].max().item())

# print("Validation image range:",
#       val_subset[0][0].min().item(),
#       val_subset[0][0].max().item())


train_loader = DataLoader(
    train_subset,
    batch_size=128,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_subset,
    batch_size=128,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    transform=test_transform,
    download=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

for name, loader in [
    ("Train", train_loader),
    ("Validation", val_loader),
    ("Test", test_loader)
]:
    images, labels = next(iter(loader))
    print(f"{name}: {images.shape}, {labels.shape}")