import torch
import torch.distributed as dist

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
print("Distributed available:", dist.is_available())
print("NCCL available:", dist.is_nccl_available())

for i in range(torch.cuda.device_count()):
    print(f"GPU {i}:", torch.cuda.get_device_name(i))