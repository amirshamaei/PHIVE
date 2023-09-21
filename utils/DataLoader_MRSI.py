import torch
from torch.utils.data import Dataset, DataLoader


class MRSI_Dataset(Dataset):
    def __init__(self,data):

        # initialize dataset
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample.unsqueeze(0)