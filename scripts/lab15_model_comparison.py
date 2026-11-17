import os
import sys
import csv
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, Subset, random_split

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, PROJECT_ROOT)

from models import create_model


# =========================
# Configuration
# =========================

SEED = 42
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0

NUM_WORKERS = 4
NUM_CLASSES = 10

MODEL_NAMES = [
    "simple_cnn",
    "resnet18",
    "mobilenet_v2",
]

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "lab15")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# =========================
# Reproducibility
# =========================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================
# Dataset
# =========================

def build_dataloaders():

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_base = torchvision.datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=False,
        transform=train_transform,
    )

    val_base = torchvision.datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=False,
        transform=eval_transform,
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=False,
        transform=eval_transform,
    )

    # 固定随机种子，划分 45000 / 5000
    train_subset, val_subset = random_split(
        range(len(train_base)),
        [45000, 5000],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_dataset = Subset(
        train_base,
        train_subset.indices,
    )

    val_dataset = Subset(
        val_base,
        val_subset.indices,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )

    return train_loader, val_loader, test_loader


# =========================
# Train one epoch
# =========================

def train_one_epoch(model, loader, criterion, optimizer, device):

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (
            outputs.argmax(dim=1) == labels
        ).sum().item()

        total_samples += batch_size

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


# =========================
# Evaluation
# =========================

def evaluate(model, loader, criterion, device):

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.inference_mode():

        for images, labels in loader:

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()

            total_samples += batch_size

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


# =========================
# Inference benchmark
# =========================

def measure_inference_throughput(
    model,
    device,
    batch_size=128,
    warmup=20,
    iterations=100,
):

    model.eval()

    dummy_input = torch.randn(
        batch_size,
        3,
        32,
        32,
        device=device,
    )

    with torch.inference_mode():

        # Warmup
        for _ in range(warmup):
            model(dummy_input)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        start_time = time.perf_counter()

        for _ in range(iterations):
            model(dummy_input)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        elapsed = time.perf_counter() - start_time

    throughput = batch_size * iterations / elapsed

    return throughput


# =========================
# CSV utilities
# =========================

def save_history(history, model_name):

    path = os.path.join(
        OUTPUT_DIR,
        f"{model_name}_history.csv",
    )

    fieldnames = list(history[0].keys())

    with open(path, "w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(history)


def save_results(results):

    path = os.path.join(
        OUTPUT_DIR,
        "results.csv",
    )

    fieldnames = list(results[0].keys())

    with open(path, "w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)


# =========================
# Main experiment
# =========================

def main():

    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    train_loader, val_loader, test_loader = (
        build_dataloaders()
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples:   {len(val_loader.dataset)}")
    print(f"Test samples:  {len(test_loader.dataset)}")

    results = []

    for model_name in MODEL_NAMES:

        print("\n" + "=" * 60)
        print(f"Training model: {model_name}")
        print("=" * 60)

        # 每个模型使用相同的随机种子
        set_seed(SEED)

        model = create_model(
            model_name,
            num_classes=NUM_CLASSES,
        ).to(device)

        total_params = sum(
            p.numel() for p in model.parameters()
        )

        trainable_params = sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=LEARNING_RATE,
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY,
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.1,
        )

        checkpoint_path = os.path.join(
            CHECKPOINT_DIR,
            f"{model_name}_best.pth",
        )

        best_val_acc = 0.0
        best_epoch = 0
        history = []

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)

        training_start = time.perf_counter()

        for epoch in range(1, EPOCHS + 1):

            train_loss, train_acc = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
            )

            val_loss, val_acc = evaluate(
                model,
                val_loader,
                criterion,
                device,
            )

            scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]

            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": current_lr,
            })

            if val_acc > best_val_acc:

                best_val_acc = val_acc
                best_epoch = epoch

                torch.save(
                    {
                        "model_name": model_name,
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "val_acc": val_acc,
                    },
                    checkpoint_path,
                )

            print(
                f"Epoch [{epoch:02d}/{EPOCHS}] "
                f"Train Loss: {train_loss:.4f} "
                f"Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} "
                f"Val Acc: {val_acc:.4f} | "
                f"LR: {current_lr:.6f}"
            )

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        training_time = (
            time.perf_counter() - training_start
        )

        peak_memory_mb = (
            torch.cuda.max_memory_allocated(device)
            / (1024 ** 2)
            if device.type == "cuda"
            else 0.0
        )

        save_history(history, model_name)

        # 加载验证集表现最好的 checkpoint
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        # 最终测试集评估
        test_loss, test_acc = evaluate(
            model,
            test_loader,
            criterion,
            device,
        )

        # GPU forward throughput，不包含数据加载
        inference_throughput = measure_inference_throughput(
            model,
            device,
            batch_size=BATCH_SIZE,
        )

        checkpoint_size_mb = (
            os.path.getsize(checkpoint_path)
            / (1024 ** 2)
        )

        result = {
            "model": model_name,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "training_time_sec": training_time,
            "inference_throughput_img_sec": inference_throughput,
            "checkpoint_size_mb": checkpoint_size_mb,
            "peak_gpu_memory_mb": peak_memory_mb,
        }

        results.append(result)
        save_results(results)

        print("\nModel summary:")
        for key, value in result.items():
            print(f"{key}: {value}")

        del model
        del optimizer
        del scheduler

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("All experiments completed.")
    print("=" * 60)

    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
