import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from models.resnet import resnet18


def train_one_epoch(
    model, train_loader, criterion,
    optimizer, scaler, device
):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16
        ):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        predicted = outputs.argmax(dim=1)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (predicted == labels).sum().item()
        total_samples += batch_size

    # 汇总所有 GPU 的训练指标
    stats = torch.tensor(
        [total_loss, total_correct, total_samples],
        dtype=torch.float64,
        device=device
    )

    dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    avg_loss = (stats[0] / stats[2]).item()
    accuracy = (stats[1] / stats[2] * 100).item()

    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, data_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        predicted = outputs.argmax(dim=1)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (predicted == labels).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples * 100

    return avg_loss, accuracy


def main():
    # 1. 初始化 DDP
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    dist.init_process_group(backend="nccl")

    writer = None

    try:
        # 2. 数据预处理
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616)
            )
        ])

        eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616)
            )
        ])

        # 3. 加载数据集
        train_dataset = datasets.CIFAR10(
            root="./data",
            train=True,
            transform=train_transform,
            download=False
        )

        val_dataset = datasets.CIFAR10(
            root="./data",
            train=True,
            transform=eval_transform,
            download=False
        )

        test_dataset = datasets.CIFAR10(
            root="./data",
            train=False,
            transform=eval_transform,
            download=False
        )

        # 4. 划分 Train / Validation
        generator = torch.Generator().manual_seed(42)

        indices = torch.randperm(
            len(train_dataset),
            generator=generator
        ).tolist()

        train_indices = indices[:45000]
        val_indices = indices[45000:]

        train_subset = Subset(train_dataset, train_indices)
        val_subset = Subset(val_dataset, val_indices)

        # 5. 分布式训练采样器
        train_sampler = DistributedSampler(
            train_subset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )

        train_loader = DataLoader(
            train_subset,
            batch_size=128,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True
        )

        # 验证集和测试集只由 Rank 0 负责评估
        if rank == 0:
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

        # 6. 创建模型并包装 DDP
        model = resnet18(num_classes=10).to(device)

        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank]
        )

        # 7. Loss / Optimizer / Scheduler / AMP
        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=0
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.1
        )

        scaler = torch.amp.GradScaler("cuda")

        # 8. 训练配置
        epochs = 30
        best_acc = 0.0
        best_epoch = 0

        os.makedirs("outputs", exist_ok=True)

        if rank == 0:
            writer = SummaryWriter(
                log_dir="outputs/runs/resnet18_ddp"
            )

        # 确保所有进程完成初始化
        dist.barrier()

        # 9. 训练循环
        for epoch in range(epochs):
            train_sampler.set_epoch(epoch)

            train_loss, train_acc = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scaler,
                device
            )

            # 只有 Rank 0 执行完整验证集评估
            if rank == 0:
                val_loss, val_acc = evaluate(
                    model.module,
                    val_loader,
                    criterion,
                    device
                )

                if val_acc > best_acc:
                    best_acc = val_acc
                    best_epoch = epoch + 1

                    torch.save(
                        model.module.state_dict(),
                        "outputs/resnet18_ddp_best.pth"
                    )

                current_lr = optimizer.param_groups[0]["lr"]

                writer.add_scalar(
                    "Loss/train", train_loss, epoch
                )
                writer.add_scalar(
                    "Loss/validation", val_loss, epoch
                )
                writer.add_scalar(
                    "Accuracy/train", train_acc, epoch
                )
                writer.add_scalar(
                    "Accuracy/validation", val_acc, epoch
                )
                writer.add_scalar(
                    "Learning_Rate", current_lr, epoch
                )

                print(
                    f"Epoch [{epoch + 1}/{epochs}] "
                    f"LR: {current_lr:.6f} "
                    f"Train Loss: {train_loss:.4f} "
                    f"Train Acc: {train_acc:.2f}% "
                    f"Val Loss: {val_loss:.4f} "
                    f"Val Acc: {val_acc:.2f}%"
                )

            # 将验证指标广播给所有进程
            # 保证各个进程同步完成当前 Epoch
            metrics = torch.zeros(
                2, dtype=torch.float64, device=device
            )

            if rank == 0:
                metrics[0] = val_loss
                metrics[1] = val_acc

            dist.broadcast(metrics, src=0)

            scheduler.step()

        if writer is not None:
            writer.close()

        # 10. 所有进程等待训练结束
        dist.barrier()

        # 11. 只由 Rank 0 加载最佳模型并测试
        if rank == 0:
            state_dict = torch.load(
                "outputs/resnet18_ddp_best.pth",
                map_location=device,
                weights_only=True
            )

            model.module.load_state_dict(state_dict)

            test_loss, test_acc = evaluate(
                model.module,
                test_loader,
                criterion,
                device
            )

            print("\nTraining finished!")
            print(f"Best Epoch: {best_epoch}")
            print(f"Best Validation Acc: {best_acc:.2f}%")
            print(f"Test Loss: {test_loss:.4f}")
            print(f"Test Accuracy: {test_acc:.2f}%")

        dist.barrier()

    finally:
        if writer is not None:
            writer.close()

        dist.destroy_process_group()


if __name__ == "__main__":
    main()