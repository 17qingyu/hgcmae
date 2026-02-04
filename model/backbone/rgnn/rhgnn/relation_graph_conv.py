import torch
from torch import nn

import dgl
from dgl.ops import edge_softmax
import dgl.function as fn


class RelationGraphConv(nn.Module):

    def __init__(self, in_dims: tuple, out_dim: int, num_heads: int, dropout: float = 0.0, negative_slope: float = 0.2):
        """
        Relation graph convolution layer
        Parameters
        ----------
        in_dims : pair of ints, input feature size
        out_dim : int, output feature size
        num_heads : int, number of heads in Multi-Head Attention
        dropout : float, optional, dropout rate, defaults: 0
        negative_slope : float, optional, negative slope rate, defaults: 0.2
        """
        super(RelationGraphConv, self).__init__()
        self._src_dim, self._dst_dim = in_dims[0], in_dims[1]
        self._out_dim = out_dim
        self._num_heads = num_heads

        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU()
        self.relu = nn.ReLU()

    def forward(self,
                graph: dgl.DGLHeteroGraph, feat: tuple,
                dst_trans_weight: nn.Parameter,
                src_trans_weight: nn.Parameter,
                rel_emb: torch.Tensor,
                rel_trans_weight: nn.Parameter):
        r"""

        Parameters
        ----------
        graph : specific relational DGLHeteroGraph
        feat : pair of torch.Tensor
            The pair contains two tensors of shape (N_{in}, D_{in_{src}})` and (N_{out}, D_{in_{dst}}).
        dst_trans_weight: Parameter (input_dst_dim, n_heads * hidden_dim)
        src_trans_weight: Parameter (input_src_dim, n_heads * hidden_dim)
        rel_emb: torch.Tensor, (relation_input_dim)
        rel_trans_weight: Parameter (relation_input_dim, n_heads * 2 * hidden_dim)

        Returns
        -------
        torch.Tensor, shape (N, H, D_out)` where H is the number of heads, and D_out is size of output feature.
        """
        graph = graph.local_var()
        # Tensor, (N_src, input_src_dim)
        feat_src = self.dropout(feat[0])
        # Tensor, (N_dst, input_dst_dim)
        feat_dst = self.dropout(feat[1])
        # Tensor, (N_src, n_heads, hidden_dim) -> (N_src, input_src_dim) * (input_src_dim, n_heads * hidden_dim)
        feat_src = torch.matmul(feat_src, src_trans_weight).view(-1, self._num_heads, self._out_dim)
        # Tensor, (N_dst, n_heads, hidden_dim) -> (N_dst, input_dst_dim) * (input_dst_dim, n_heads * hidden_dim)
        feat_dst = torch.matmul(feat_dst, dst_trans_weight).view(-1, self._num_heads, self._out_dim)
        # Tensor, (n_heads, 2 * hidden_dim) -> (1, input_dst_dim) * (input_dst_dim, n_heads * hidden_dim)
        rel_attn_weight = torch.matmul(rel_emb.unsqueeze(dim=0), rel_trans_weight).view(self._num_heads, 2 * self._out_dim)

        # first decompose the weight vector into [a_l || a_r], then
        # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j, This implementation is much efficient
        # Tensor, (N_dst, n_heads, 1),   (N_dst, n_heads, hidden_dim) * (n_heads, hidden_dim)
        e_dst = (feat_dst * rel_attn_weight[:, :self._out_dim]).sum(dim=-1, keepdim=True)
        # Tensor, (N_src, n_heads, 1),   (N_src, n_heads, hidden_dim) * (n_heads, hidden_dim)
        e_src = (feat_src * rel_attn_weight[:, self._out_dim:]).sum(dim=-1, keepdim=True)
        # (N_src, n_heads, hidden_dim), (N_src, n_heads, 1)
        graph.srcdata.update({'ft': feat_src, 'e_src': e_src})
        # (N_dst, n_heads, 1)
        graph.dstdata.update({'e_dst': e_dst})
        # compute edge attention, e_src and e_dst are a_src * Wh_src and a_dst * Wh_dst respectively.
        graph.apply_edges(fn.u_add_v('e_src', 'e_dst', 'e'))
        # shape (edges_num, heads, 1)
        e = self.leaky_relu(graph.edata.pop('e'))

        # compute softmax
        graph.edata['a'] = edge_softmax(graph, e)

        graph.update_all(fn.u_mul_e('ft', 'a', 'msg'), fn.sum('msg', 'feat'))
        # (N_dst, n_heads * hidden_dim), reshape (N_dst, n_heads, hidden_dim)
        dst_feat = graph.dstdata.pop('feat').reshape(-1, self._num_heads * self._out_dim)

        dst_feat = self.relu(dst_feat)

        return dst_feat
