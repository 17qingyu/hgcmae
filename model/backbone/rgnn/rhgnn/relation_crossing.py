import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationCrossing(nn.Module):

    def __init__(self, in_feats: int, out_feats: int, num_heads: int, dropout: float = 0.0,
                 negative_slope: float = 0.2):
        """
        Relation crossing layer
        Parameters
        ----------
        in_feats : pair of ints, input feature size
        out_feats : int, output feature size
        num_heads : int, number of heads in Multi-Head Attention
        dropout : float, optional, dropout rate, defaults: 0.0
        negative_slope : float, optional, negative slope rate, defaults: 0.2
        """
        super(RelationCrossing, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._num_heads = num_heads

        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self, dst_node_features: torch.Tensor, rel_cross_attn_weight: nn.Parameter):
        """
        :param dst_node_features: a tensor of (dst_type_node_relations_num, num_dst_nodes, n_heads * hidden_dim)
        :param rel_cross_attn_weight: Parameter the shape is (n_heads, hidden_dim)
        :return: output_features: a Tensor
        """
        if len(dst_node_features) == 1:
            # (num_dst_nodes, n_heads * hidden_dim)
            dst_node_features = dst_node_features.squeeze(dim=0)
        else:
            # (dst_type_node_relations_num, num_dst_nodes, n_heads, hidden_dim)
            dst_node_features = dst_node_features.reshape(dst_node_features.shape[0], -1, self._num_heads,
                                                          self._out_feats)
            # shape -> (dst_type_node_relations_num, dst_nodes_num, n_heads, 1),
            # (dst_type_node_relations_num, dst_nodes_num, n_heads, hidden_dim) * (n_heads, hidden_dim)
            dst_node_rel_atten = ((dst_node_features * rel_cross_attn_weight)
                                  .sum(dim=-1, keepdim=True))
            dst_node_rel_atten = F.softmax(self.leaky_relu(dst_node_rel_atten), dim=0)
            # shape -> (dst_nodes_num, n_heads, hidden_dim),
            # (dst_type_node_relations_num, dst_nodes_num, n_heads, hidden_dim) *
            # (dst_type_node_relations_num, dst_nodes_num, n_heads, 1)
            dst_node_features = (dst_node_features * dst_node_rel_atten).sum(dim=0)
            dst_node_features = self.dropout(dst_node_features)
            # shape -> (dst_nodes_num, n_heads * hidden_dim)
            dst_node_features = dst_node_features.reshape(-1, self._num_heads * self._out_feats)

        return dst_node_features
