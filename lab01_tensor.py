import torch

rand_tensor = torch.rand((3, 4))
ones_tensor = torch.ones((3, 4))
ans = rand_tensor + ones_tensor
print(ans.shape, ans.dtype, ans.device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ans = ans.to(device)
print(ans.device)