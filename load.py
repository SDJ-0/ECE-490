import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class AdditionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return AdditionDataset(self.X[idx], self.y[idx])
        return self.X[idx], self.y[idx]


def addition_task(filename, batch_size, train_num, test_num):
    data = np.loadtxt(filename, delimiter=',').astype(np.float32)
    x = data[:, 1:]
    y = data[:, 0]
    X = torch.tensor(x.reshape((x.shape[0], x.shape[1] // 2, 2)))
    Y = torch.tensor(y.reshape((y.shape[0], 1)))
    addition_dataset = AdditionDataset(X, Y)

    train_data_loader = DataLoader(addition_dataset[:train_num], batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(addition_dataset[train_num:train_num + test_num], batch_size=batch_size, shuffle=True)

    return train_data_loader, test_data_loader


class CopyDataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return CopyDataset(self.X[idx])
        return self.X[idx], self.X[idx, :10]


def copy_task(filename, batch_size, train_num, test_num):
    data = np.loadtxt(filename, delimiter=',')
    data = data.reshape((data.shape[0], -1, 10))
    X = torch.tensor(data, dtype=torch.float32)
    copy_dataset = CopyDataset(X)

    train_data_loader = DataLoader(copy_dataset[:train_num], batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(copy_dataset[train_num:train_num + test_num], batch_size=batch_size, shuffle=True)

    return train_data_loader, test_data_loader


if __name__ == '__main__':
    copy_task('data/Copy_task/data/data100', 32, 10000, 1000)
