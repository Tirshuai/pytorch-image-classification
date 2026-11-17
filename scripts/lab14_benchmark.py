
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)


import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

from models.resnet import resnet18


def train_one_epoch(
    model, loader, criterion, optimizer, scaler, device
):
    model.train()

    for images, labels in loader:
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


def main():
    # 1. 判断当前运行模式
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    is_ddp = world_size > 1

    if is_ddp:
        dist.init_process_group(backend="nccl")

    try:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        # 2. 加载 CIFAR-10
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616)
            )
        ])

        dataset = datasets.CIFAR10(
            root="./data",
            train=True,
            transform=transform,
            download=False
        )

        generator = torch.Generator().manual_seed(42)

        indices = torch.randperm(
            len(dataset),
            generator=generator
        ).tolist()

        train_indices = indices[:45000]
        train_dataset = Subset(dataset, train_indices)

        # 3. 创建 Sampler 和 DataLoader
        global_batch_size = 512
        batch_size = global_batch_size // world_size

        if is_ddp:
            sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True
            )
        else:
            sampler = None

        loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=4,
            pin_memory=True
        )

        # 4. 创建 ResNet18
        model = resnet18(num_classes=10).to(device)

        if is_ddp:
            model = DDP(
                model,
                device_ids=[local_rank]
            )

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.01,
            momentum=0.9
        )

        scaler = torch.amp.GradScaler("cuda")

        # 5. 预热
        warmup_epochs = 2
        benchmark_epochs = 5

        for epoch in range(warmup_epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)

            train_one_epoch(
                model, loader, criterion,
                optimizer, scaler, device
            )

        # 6. 正式性能测试
        torch.cuda.reset_peak_memory_stats(device)

        epoch_times = []

        for epoch in range(benchmark_epochs):
            if sampler is not None:
                sampler.set_epoch(warmup_epochs + epoch)

            # 确保所有进程准备好开始
            if is_ddp:
                dist.barrier()

            # 等待 GPU 完成之前的任务
            torch.cuda.synchronize(device)

            start_time = time.perf_counter()

            train_one_epoch(
                model, loader, criterion,
                optimizer, scaler, device
            )

            # 等待当前 GPU 的任务全部完成
            torch.cuda.synchronize(device)

            local_time = time.perf_counter() - start_time

            # 取所有 Rank 中较慢的耗时
            if is_ddp:
                time_tensor = torch.tensor(
                    local_time,
                    dtype=torch.float64,
                    device=device
                )

                dist.all_reduce(
                    time_tensor,
                    op=dist.ReduceOp.MAX
                )

                epoch_time = time_tensor.item()
            else:
                epoch_time = local_time

            epoch_times.append(epoch_time)

            if rank == 0:
                throughput = len(train_dataset) / epoch_time

                print(
                    f"Benchmark Epoch [{epoch + 1}/{benchmark_epochs}] "
                    f"Time: {epoch_time:.2f}s "
                    f"Throughput: {throughput:.2f} images/s"
                )

        # 7. 统计平均耗时和吞吐量
        avg_epoch_time = sum(epoch_times) / len(epoch_times)

        avg_throughput = (
            len(train_dataset) / avg_epoch_time
        )

        # 8. 统计峰值显存
        peak_memory = torch.cuda.max_memory_allocated(device)

        if is_ddp:
            memory_tensor = torch.tensor(
                peak_memory,
                dtype=torch.float64,
                device=device
            )

            dist.all_reduce(
                memory_tensor,
                op=dist.ReduceOp.MAX
            )

            peak_memory = memory_tensor.item()

        peak_memory_gb = peak_memory / (1024 ** 3)

        if rank == 0:
            print("\n========== Benchmark Result ==========")
            print(f"GPU Count: {world_size}")
            print(f"Global Batch Size: {global_batch_size}")
            print(f"Average Epoch Time: {avg_epoch_time:.2f}s")
            print(f"Average Throughput: {avg_throughput:.2f} images/s")
            print(f"Peak GPU Memory: {peak_memory_gb:.2f} GB")
            print("======================================")

    finally:
        if is_ddp:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
