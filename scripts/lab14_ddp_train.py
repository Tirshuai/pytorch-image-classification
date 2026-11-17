import os
import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])

torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

dist.init_process_group(backend="nccl")

model = torch.nn.Sequential(
    torch.nn.Linear(10, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 2)
).to(device)

model = torch.nn.parallel.DistributedDataParallel(
    model, 
    device_ids=[local_rank]
)

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

x = torch.randn(1000, 10)
y = torch.randint(0, 2, (1000, ))

dataset = TensorDataset(x, y)

sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True
)

loader = DataLoader(
    dataset,
    batch_size=32,
    sampler=sampler
)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

model.train()

epoch_num = 3

for epoch in range(epoch_num):
    sampler.set_epoch(epoch)

    for images, labels in loader:
        images = images.to(device, non_blocking = True)
        labels = labels.to(device, non_blocking = True)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        print(f"Rank {rank} Epoch {epoch + 1} Loss: {loss}")

for name, param in model.module.named_parameters():
    param_copy = param.detach().clone()

    dist.all_reduce(
        param_copy,
        op=dist.ReduceOp.SUM
    )

    param_avg = param_copy / world_size

    diff = (param - param_avg).abs().max().item()

    print(f"Rank {rank}, {name}, Max Diff: {diff:.10f}")

dist.destroy_process_group()