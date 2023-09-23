import math

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class MRSI_Dataset(Dataset):
    def __init__(self,data,engine):
        self.engine = engine
        # initialize dataset
        self.data = data
        self.t = torch.from_numpy(self.engine.t[0:data.shape[1]].T).float()
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.engine.parameters['aug_params']:
            sample = self.get_augment(sample,
                                      self.engine.parameters['aug_params'][0],
                                      self.engine.parameters['aug_params'][1],
                                      self.engine.parameters['aug_params'][2],
                                      self.engine.parameters['aug_params'][3])
        else:
            sample = sample.unsqueeze(0)
        return sample

    def get_augment(self, signal, f_band, ph_band, d_band, noise_level):

        shift = f_band * torch.rand(1) - (f_band / 2)
        ph = ph_band * torch.rand(1) * math.pi - ((ph_band / 2) * math.pi)
        d = torch.rand(1) * d_band

        freq = -2 * math.pi * shift

        y = signal * torch.exp(1j * (ph + freq * self.t))  # t is the time vector
        y = y * torch.exp(-d * self.t)

        noise = (torch.randn(1, len(signal)) +
                 1j * torch.randn(1, len(signal))) * noise_level  # example noise level
        return y + noise