import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_channels: int, hid_channels: int, out_channels: int, bias=True):
        super().__init__()
        self.linear1 = nn.Linear(in_channels, hid_channels, bias=bias)
        self.linear2 = nn.Linear(hid_channels, out_channels, bias=bias)
        self.relu = nn.PReLU()

    def forward(self, x):
        h = self.relu(self.linear1(x))
        h = self.linear2(h)
        return h
