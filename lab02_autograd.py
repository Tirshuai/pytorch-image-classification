
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class MyDataset(Dataset):

    def __init__(self):
        # 创建 100 个样本
        # 每个样本包含 4 个特征
        self.data = torch.randn(100, 4)

        # 创建 100 个标签
        # 标签取值为 0 或 1
        self.labels = torch.randint(0, 2, (100,))

    def __len__(self):
        return self.data.size(0)
    def __getitem__(self, idx):
        return (self.data[idx], self.labels[idx])


if __name__ == "__main__":
    dataset = MyDataset()

    print("Dataset length:", len(dataset))

    sample, label = dataset[0]

    print("Sample:", sample)
    print("Sample shape:", sample.shape)
    print("Label:", label)

    loader = DataLoader(
        dataset,
        batch_size = 16,
        shuffle = True,
        num_workers = 0,
        drop_last = False
    )

    cnt = 0
    for batch_data, batch_labels in loader:
        print(batch_data)
        print(batch_labels)
        cnt += 1
    print(cnt)