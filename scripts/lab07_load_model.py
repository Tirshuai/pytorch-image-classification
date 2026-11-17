from engine.train import train_one_epoch
from engine.evaluate import evaluate

import torch
import torch.nn as nn
import torch.optim as optim

from models.simple_cnn import SimpleCNN

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 10

model = SimpleCNN(num_classes).to(device)

state_dict = torch.load(
    "outputs/best_model.pth",
    map_location=device,
    weights_only=True
)

model.load_state_dict(state_dict)

criterion = nn.CrossEntropyLoss()

model.eval()

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

test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)

print(f"test acc : {test_accuracy:.2f}")