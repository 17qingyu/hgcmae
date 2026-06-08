from typing import Union

from dgl import DGLHeteroGraph
import torch
from torch import Tensor
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(
        self,
        rgnn: nn.Module,
        project: nn.Module,
        checkpoint_path: Union[None, str] = None
    ):
        super(Encoder, self).__init__()
        if rgnn is None or project is None:
            # 如果没有提供子模块，也许需要抛出异常或创建默认的子模块
            raise ValueError(
                "rgnn, mask, project must be provided if no default is defined")
        self.project = project
        self.rgnn = rgnn

        # 如果指定了 checkpoint_path，则加载预训练权重
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.load_state_dict(checkpoint['state_dict'])

    def forward(self, g: DGLHeteroGraph, x_dict: dict[str, Tensor]):
        project_x = self.project(x_dict)
        h_dict = self.rgnn(g, project_x)

        return h_dict


class Decoder(nn.Module):
    def __init__(
        self,
        rgnn: nn.Module,
        rev_project: nn.Module,
    ):
        super().__init__()
        self.rgnn = rgnn
        self.rev_project = rev_project
        
    def forward(self, g: DGLHeteroGraph, x_dict: dict[str, Tensor]):
        h_dict = self.rgnn(g, x_dict)
        res_feat = self.rev_project(h_dict)
        
        return res_feat
    