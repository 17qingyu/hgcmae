from typing import Iterable, Mapping

import torch
import torch.nn as nn


class FeatureMask(nn.Module):
    def __init__(
        self,
        start: float,
        step: float,
        end: float,
        mask_dim: dict
    ):
        super(FeatureMask, self).__init__()
        self.start = start
        self.step = step
        self.end = end
        self.mask_token = nn.ParameterDict(
            {ntyp: torch.zeros((dim,), dtype=torch.float32)
             for ntyp, dim in mask_dim.items()}
        )

    def get_rate(self, epoch: int) -> float:
        return min(self.start + self.step * epoch, self.end)

    def forward(
        self,
        x_dict: Mapping[str, torch.Tensor],
        epoch: int,
        target_type: str
    ) -> tuple[Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]]:
        x_clone = {ntyp: x.clone() for ntyp, x in x_dict.items()}
        mask_h_dict = {ntyp: x.clone() for ntyp, x in x_dict.items()}
        
        num_nodes = {ntyp: x.shape[0] for ntyp, x in x_clone.items()}
        num_mask_nodes = {
            ntyp: int(self.get_rate(epoch) * num_node)
            for ntyp, num_node in num_nodes.items()
        }
        
        mask_nodes_s = {ntyp: torch.tensor([], dtype=torch.long) for ntyp in num_nodes}
        mask_nodes_t = dict()
        for ntyp in num_nodes:
            perm = torch.randperm(num_nodes[ntyp], device=x_clone[ntyp].device)
            mask_node = perm[:num_mask_nodes[ntyp]]
            mask_h_dict[ntyp][mask_node] = self.mask_token[ntyp]
            mask_nodes_t[ntyp] = mask_node

        x_clone[target_type], mask_h_dict[target_type] = mask_h_dict[target_type], x_clone[target_type],
        mask_nodes_s[target_type], mask_nodes_t[target_type] = mask_nodes_t[target_type], mask_nodes_s[target_type] 

        return x_clone, mask_h_dict, mask_nodes_s, mask_nodes_t
