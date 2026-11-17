
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms


def main():
    # 1. 初始化分布式环境
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    dist.init_process_group(backend="nccl")

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

        # 3. 创建训练集和验证集
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

        # 4. 使用固定随机种子划分数据
        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(
            len(train_dataset),
            generator=generator
        ).tolist()

        train_indices = indices[:45000]
        val_indices = indices[45000:]

        train_subset = Subset(train_dataset, train_indices)
        val_subset = Subset(val_dataset, val_indices)

        # 5. 创建 DistributedSampler
        train_sampler = DistributedSampler(
            train_subset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )

        # 6. 创建 DataLoader
        train_loader = DataLoader(
            train_subset,
            batch_size=128,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True
        )

        # 验证集暂时不切分，后续训练阶段再处理评估
        print(
            f"Rank {rank} | GPU {local_rank} | "
            f"Train samples: {len(train_sampler)} | "
            f"Validation samples: {len(val_subset)} | "
            f"Train batches: {len(train_loader)}"
        )

        # 7. 检查一个 batch
        train_sampler.set_epoch(0)

        images, labels = next(iter(train_loader))
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        print(
            f"Rank {rank} | "
            f"Images: {images.shape} | "
            f"Labels: {labels.shape} | "
            f"Device: {images.device}"
        )

        # 8. 检查两个进程都完成数据加载
        dist.barrier()

        if rank == 0:
            print("DDP CIFAR-10 data loading test passed!")

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
