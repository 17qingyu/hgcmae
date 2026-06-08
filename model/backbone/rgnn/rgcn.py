import dgl
import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F

class RGCN(nn.Module):
    def __init__(
            self, etypes: list, ntypes: list, in_channels: int,  # 新增ntypes参数
            hid_channels: int, num_layers: int, dropout: float = 0.,
            use_norm = True
    ):
        super().__init__()
        self.ntypes = ntypes  # 节点类型列表
        
        # 定义各层GraphConv和对应的LayerNorm
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()  # 每层独立的LayerNorm
        
        # 输入层
        self.layers.append(
            self._build_hetero_conv(etypes, in_channels, hid_channels, activation=True)
        )
        self.use_norm = use_norm
        if self.use_norm:
            self.layer_norms.append(self._build_layer_norms(ntypes, hid_channels))
        
        # 中间层
        for _ in range(num_layers-2):
            self.layers.append(
                self._build_hetero_conv(etypes, hid_channels, hid_channels, activation=True)
            )
            if self.use_norm:
                self.layer_norms.append(self._build_layer_norms(ntypes, hid_channels))
        
        # 输出层（无激活函数）
        self.layers.append(
            self._build_hetero_conv(etypes, hid_channels, hid_channels, activation=False)
        )
        if self.use_norm:
            self.layer_norms.append(self._build_layer_norms(ntypes, hid_channels))
        
        self.dropout = nn.Dropout(dropout)
        self.att_proj = nn.Linear(hid_channels, 1)  # 修复缺失的att_proj定义

    def _build_hetero_conv(self, etypes, in_feat, out_feat, activation=False):
        """创建HeteroGraphConv层"""
        conv_dict = {}
        for etype in etypes:
            activation_fn = torch.relu if activation else None
            conv_dict[etype] = dglnn.GraphConv(
                in_feat, out_feat, norm="both", activation=activation_fn
            )
        return dglnn.HeteroGraphConv(conv_dict, aggregate="sum")
    
    def _build_layer_norms(self, ntypes, hid_channels):
        """为每个节点类型创建独立的LayerNorm"""
        return nn.ModuleDict({ntype: nn.LayerNorm(hid_channels) for ntype in ntypes})

    def forward(self, hg, x):
        h = x
        for layer, layer_norm in zip(self.layers, self.layer_norms):
            h = layer(hg, h)
            # 对每个节点类型应用LayerNorm
            if self.use_norm:
                for ntype in self.ntypes:
                    h[ntype] = layer_norm[ntype](h[ntype])
            # 中间层后应用Dropout（最后一层不应用）
            if layer != self.layers[-1]:
                for ntype in h:
                    h[ntype] = self.dropout(h[ntype])
        return h

    # def aggregate_relations(self, outputs, dst_type):
    #     """带注意力的关系聚合（修复att_proj未定义问题）"""
    #     stacked = torch.stack(outputs, dim=0)  # (R, N, F)
    #     scores = self.att_proj(stacked).squeeze(-1)  # (R, N)
    #     attn_weights = F.softmax(scores, dim=0)
    #     aggregated = (stacked * attn_weights.unsqueeze(-1)).sum(dim=0)
    #     return aggregated
    