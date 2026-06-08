import torch
import torch.nn as nn


class Project(nn.Module):
    def __init__(self, in_channels, out_channel: int):
        super().__init__()
        self.lin_dict = torch.nn.ModuleDict()
        for ntyp, in_channel in in_channels.items():
            self.lin_dict[ntyp] = nn.Linear(in_channel, out_channel)

    def forward(self, x_dict):
        project_x_dict = dict()
        for node_type, x in x_dict.items():
            project_x_dict[node_type] = self.lin_dict[node_type](x)
        return project_x_dict


class ReverseProject(nn.Module):
    def __init__(self, in_channels,  out_channels: dict[str, int]):
        super().__init__()
        self.lin_dict = torch.nn.ModuleDict()
        for ntyp, out_channel in out_channels.items():
            self.lin_dict[ntyp] = nn.Linear(in_channels, out_channel)

    def forward(self, x_dict):
        project_x_dict = dict()
        for node_type, x in x_dict.items():
            project_x_dict[node_type] = self.lin_dict[node_type](x)
        return project_x_dict
