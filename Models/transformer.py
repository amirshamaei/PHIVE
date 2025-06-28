import math

import torch
from torch import nn


class Transformer(torch.nn.Module):
    def __init__(self, insize, outsize, in_channel=1):
        super().__init__()
        d_model = 256
        n_letgh = int(insize/4)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dropout=0.1, activation='gelu')
        self.stem = torch.nn.Sequential(torch.nn.Conv1d(in_channel, int(d_model), 5,  stride=5, padding=2,bias=False))
                                        # torch.nn.Conv1d(int(d_model/2), d_model, 2, stride=2, padding=2,bias=False))

        self.encoder = torch.nn.TransformerEncoder(encoder_layer,num_layers=1)
        self.reg = torch.nn.Sequential(torch.nn.Flatten(),torch.nn.LazyLinear(outsize))
        # self.pos = torch.nn.Parameter(torch.randn((1,len,dim)))
        self.pos2 = PositionalEncoding(d_model, 256)
    def forward(self,x):
        # x = x.permute(0, 1, 2)
        x = self.stem(x)
        x = x.permute(2, 0, 1)
        # x = self.pos2(x)
        # x = x + self.pos
        x1 = self.encoder(x)
        x = x1.mean(0)

        x= self.reg(x)

        return x,x1


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x1 = x + self.pe[:, :x.size(1)]
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionalStemBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvolutionalStemBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=32, stride=16, padding=0)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, stride=3, padding=1)
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=7, stride=5, padding=1)
        self.norm = nn.GroupNorm(8,out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x)
        # x = self.conv2(x)
        # x = F.leaky_relu(x)
        # x = self.conv3(x)
        # x = F.leaky_relu(x)
        # x = self.norm(x)
        return x

class TransformerB(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, hidden_size, num_layers,outsize):
        super(TransformerB, self).__init__()

        self.stem_block = ConvolutionalStemBlock(in_channels, hidden_size)

        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden_size, num_heads, dim_feedforward=out_channels)
            for _ in range(num_layers)
        ])
        self.reg = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.LazyLinear(outsize))

    def forward(self, x):
        x = self.stem_block(x)

        # Reshape the input for the transformer
        x = x.permute(2, 0, 1)  # (seq_len, batch_size, hidden_size)

        for layer in self.transformer_layers:
            x = layer(x)

        # Reshape the output back to the original shape
        x1 = x.permute(1, 2, 0) # (batch_size, hidden_size, seq_len)
        x = self.reg(x1)
        return x,x1
