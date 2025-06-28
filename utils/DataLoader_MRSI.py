import math
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, TensorDataset


# class MRSI_Dataset(TensorDataset):
#     tensors: Tuple[Tensor, ...]
#     def __init__(self,tensors,engine):
#         self.engine = engine
#         # initialize dataset
#         self.tensors = tensors
#         self.t = torch.from_numpy(self.engine.t[0:tensors.shape[1]].T).float()
#     def __len__(self):
#         return len(self.tensors)
#
#     def __getitem__(self, index):
#         # tuple(tensor[index] for tensor in self.tensors)
#         # sample = self.data[idx]
#         # if False:#self.engine.parameters['aug_params']:
#             sample = self.get_augment(sample,
#                                       self.engine.parameters['aug_params'][0],
#                                       self.engine.parameters['aug_params'][1],
#                                       self.engine.parameters['aug_params'][2],
#                                       self.engine.parameters['aug_params'][3])
#         # else:
#         #     sample = sample.unsqueeze(0)
#         # return sample
#
#         return tuple(tensor[index] for tensor in self.tensors)

class MRSI_Dataset(Dataset[Tuple[Tensor, ...]]):
    r"""Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Args:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """
    tensors: Tuple[Tensor, ...]

    def __init__(self, *tensors: Tensor, engine ) -> None:
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tensors
        self.engine = engine
        self.t = torch.from_numpy(self.engine.t[0:tensors[0].shape[1]].T).float()
    def __getitem__(self, index):
        return tuple(self.get_augment(tensor[index],
                      self.engine.parameters['aug_params'][0],
                      self.engine.parameters['aug_params'][1],
                      self.engine.parameters['aug_params'][2],
                      self.engine.parameters['aug_params'][3]) for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)
    def get_augment(self, signal, f_band, ph_band, d_band, noise_level):

        shift = f_band * torch.rand(1) - (f_band / 2)
        ph = ph_band * torch.rand(1) * math.pi - ((ph_band / 2) * math.pi)
        d = torch.rand(1) * d_band

        freq = -2 * math.pi * shift

        y = signal * torch.exp(1j * (ph + freq * self.t))  # t is the time vector
        y = y * torch.exp(-d * self.t)

        noise = (torch.randn(1, len(signal)) +
                 1j * torch.randn(1, len(signal))) * noise_level  # example noise level
        return (y + noise).squeeze()