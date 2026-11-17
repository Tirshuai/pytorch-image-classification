import os
import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])

torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl")

tensor = torch.tensor(
    [float(rank + 1)], 
    device=f"cuda:{local_rank}"
)

print(f"Rank {rank} before: {tensor.item()}")

dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

print(f"Rank {rank} after {tensor.item()}")
