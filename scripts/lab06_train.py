import torch
import torch.nn as nn
import torch.optim as optim

from models.simple_cnn import SimpleCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 10
model = SimpleCNN(num_classes).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9
)

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding = 4),
    transforms.RandomHorizontalFlip(p = 0.5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = (0.4914, 0.4822, 0.4465),
        std = (0.2470, 0.2435, 0.2616)
    )
])

# test_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean = (0.4914, 0.4822, 0.4465),
#         std = (0.2470, 0.2435, 0.2616)
#     )
# ])

# 训练集
train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    transform=train_transform,
    download=False
)

# # 测试集
# test_dataset = datasets.CIFAR10(
#     root="./data",
#     train=False,
#     transform=test_transform,
#     download=False
# )

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=4,
    pin_memory=True
) 

model.train()
total_loss = 0.0
total_sample = 0
batch_id = 0
total_correct = 0

for images, labels in train_loader:
    images = images.to(device, non_blocking = True)
    labels = labels.to(device, non_blocking = True)

    optimizer.zero_grad()

    outputs = model(images)

    loss = criterion(outputs, labels)

    loss.backward()
    optimizer.step()

    predicted = outputs.argmax(dim=1)
    total_correct += (predicted == labels).sum().item()

    batch_size = images.size(0)
    total_sample += batch_size
    total_loss += loss.item()*batch_size

    if (batch_id + 1) % 100 == 0:
        print(
            f"Batch [{batch_id + 1}/{len(train_loader)}], "
            f"Loss: {loss.item():.4f}"
        )
    batch_id += 1

avg_loss = total_loss/total_sample
print(f"Train Average Loss: {avg_loss:.4f}")
train_accuracy = total_correct / total_sample * 100
print(f"Train accuracy: {train_accuracy:.2f}%")


test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean = (0.4914, 0.4822, 0.4465),
        std = (0.2470, 0.2435, 0.2616)
    )
])

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

def evaluate(model, test_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            total_correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            batch_size = labels.size(0)
            total_loss += loss.item()*batch_size
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples * 100

    return avg_loss, accuracy    

test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.2f}%")