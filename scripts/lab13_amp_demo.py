import torch

device = torch.device("cuda")

# x = torch.randn(4096, 4096, device=device)

# print("FP32 dtype:", x.dtype)
# print("FP32 memory:", x.element_size() * x.numel() / 1024**2, "MB")

# x_fp16 = x.to(torch.float16)

# print("FP16 dtype:", x_fp16.dtype)
# print("FP16 memory:", x_fp16.element_size() * x_fp16.numel() / 1024**2, "MB")

a = torch.randn(1024, 1024, device=device)
b = torch.randn(1024, 1024, device=device)

print("Before autocast:")

c = a @ b
print(c.dtype)

print("Inside autocast:")

with torch.autocast(device_type="cuda", dtype=torch.float16):
    c = a @ b
    print(c.dtype)