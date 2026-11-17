from engine.train import train_one_epoch
from engine.evaluate import evaluate

import torch
import torch.nn as nn
import torch.optim as optim

from models.simple_cnn import SimpleCNN

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def train(
    model, 
    train_loader, 
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs
):
    history = {
        "train_loss" : [],
        "train_acc" : [],
        "val_loss" : [],
        "val_acc" : []
    }

    best_acc = 0.0

    for epoch in range(num_epochs):

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Learning Rate: {current_lr:.6f}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        scheduler.step()

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_acc:.2f}% "
            f"Validation Loss: {val_loss:.4f} "
            f"Validation Acc: {val_acc:.2f}%"
        )
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "outputs/best_model.pth")
            print(f"Best model saved : Validation Acc: {val_acc:.2f}")

    return history
    


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 10
model = SimpleCNN(num_classes).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=0
)

scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10,
    gamma=0.1
)

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

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
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

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

num_epochs = 30
history = train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs)

state_dict = torch.load(
    "outputs/best_model.pth",
    map_location=device,
    weights_only=True
)

model.load_state_dict(state_dict)

test_loss, test_acc = evaluate(
    model, test_loader, criterion, device
)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test acc: {test_acc:.2f}%")

import os
import matplotlib.pyplot as plt

os.makedirs("outputs", exist_ok=True)

# Loss 曲线
plt.figure()
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()
plt.grid()
plt.savefig("outputs/loss_curve.png")
plt.close()

# Accuracy 曲线
plt.figure()
plt.plot(history["train_acc"], label="Train Accuracy")
plt.plot(history["val_acc"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy Curve")
plt.legend()
plt.grid()
plt.savefig("outputs/accuracy_curve.png")
plt.close()