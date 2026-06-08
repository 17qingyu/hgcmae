import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

from .relation_graph_conv import RelationGraphConv
from .hetero_conv import HeteroGraphConv
from .relation_crossing import RelationCrossing
from .relation_fusing import RelationFusing

class RHGNN(nn.Module):
    def __init__(
            self,
            ntypes: list,
            etypes: list,
            in_dim: int,
            hid_dim: int,
            rel_in_dim: int = None,
            rel_hid_dim: int = None,
            num_layers: int = 2,
            num_heads: int = 4,
            dropout: float = 0.2,
            negative_slope: float = 0.2,
            use_res: bool = True,
            use_norm: bool = False
    ):
        """
        :param ntypes: heterogeneous graph's node types
        :param etypes: heterogeneous graph's edge types
        :param in_dim: node input dimension dictionary
        :param hid_dim: int, node hidden dimension
        :param rel_in_dim: int, relation input dimension
        :param rel_hid_dim: int, relation hidden dimension
        :param num_layers: int, number of stacked layers
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        :param negative_slope: float, negative slope
        :param use_res: boolean, residual connections or not
        :param use_norm: boolean, layer normalization or not
        """
        super().__init__()

        self.ntypes = ntypes
        # etypes = [tuple(etype) for etype in etypes]
        
        self.etypes = etypes

        self.in_dim = in_dim
        self.hid_dim = hid_dim

        if rel_in_dim is None:
            rel_in_dim = in_dim
        if rel_hid_dim is None:
            rel_hid_dim = hid_dim
            
        self.rel_in_dim = rel_in_dim
        self.rel_hid_dim = rel_hid_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.neg_slope = negative_slope
        self.use_res = use_res
        self.use_norm = use_norm

        self.rel_emb = nn.ParameterDict(
            {
                etype: nn.Parameter(torch.randn(rel_in_dim, 1))
                for etype in etypes
            }
        )

        self.layers = nn.ModuleList()

        self.layers.append(
            RHGNNLayer(
                ntypes, etypes, in_dim, hid_dim,
                rel_in_dim, rel_hid_dim, num_heads,
                dropout, negative_slope, use_res, use_norm
            )
        )
        for _ in range(1, num_layers):
            self.layers.append(
                RHGNNLayer(
                    ntypes, etypes, hid_dim * num_heads, hid_dim,
                                    rel_hid_dim * num_heads, rel_hid_dim, num_heads,
                    dropout, negative_slope, use_res, use_norm
                )
            )

        self.node_trans_weight = nn.ParameterDict({
            etype: nn.Parameter(torch.randn(num_heads, hid_dim, hid_dim))
            for etype in etypes
        })

        self.rel_trans_weight = nn.ParameterDict({
            etype: nn.Parameter(torch.randn(num_heads, rel_hid_dim, hid_dim))
            for etype in etypes
        })

        self.relation_fusing = RelationFusing(
            node_hid_dim=hid_dim,
            rel_hid_dim=rel_hid_dim,
            num_heads=num_heads,
            dropout=dropout,
            negative_slope=negative_slope
        )

        self.reset_parameters()

    @classmethod
    def build_from_args(cls, args_dict: dict):
        return cls(
            args_dict["ntypes"],
            args_dict["etypes"],
            args_dict["in_dim"],
            args_dict["hid_dim"],
            args_dict["rel_in_dim"],
            args_dict["rel_hid_dim"],
            args_dict.get("num_layers"),
            args_dict.get("num_heads", 1),
            args_dict.get("dropout", 0.),
            args_dict.get("negative_slope", 0.),
            args_dict.get("use_res", True),
            args_dict.get("use_norm", False),
        )

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')

        for etype in self.etypes:
            nn.init.xavier_normal_(self.rel_emb[etype], gain=gain)
        for etype in self.etypes:
            nn.init.xavier_normal_(self.node_trans_weight[etype], gain=gain)
        for etype in self.etypes:
            nn.init.xavier_normal_(self.rel_trans_weight[etype], gain=gain)

    def forward(self, hg, node_feat: dict, rel_emb: dict = None):
        """
        :param hg: dgl.DGLHeteroGraph
        :param node_feat: target node features under each relation, dict, {etype: features}
        :param rel_emb: embedding for each relation, dict, {etype: feature} or None
        :return:
        """

        rel_node_feat = {
            (stype, etype, dtype): node_feat[dtype]
            for stype, etype, dtype in hg.canonical_etypes
        }
        dtypes = {dtype for *_, dtype in rel_node_feat}

        if rel_emb is None:
            rel_emb = {}
            for etype in self.rel_emb:
                rel_emb[etype] = self.rel_emb[etype].flatten()

        for layer in self.layers:
            rel_node_feat, rel_emb = layer(hg, rel_node_feat, rel_emb)

        rel_fusion_emb_dict = {}
        # relation_target_node_features -> {(src_type, etype, dst_type): target_node_features}
        for dst_type in dtypes:
            dst_etypes = [etype for _, etype, dtype in rel_node_feat
                          if dtype == dst_type]
            dst_node_feat = [rel_node_feat[hg.to_canonical_etype(etype)]
                             for etype in dst_etypes]
            dst_rel_emb = [rel_emb[etype] for etype in dst_etypes]
            dst_node_trans_weight = [self.node_trans_weight[etype]
                                     for etype in dst_etypes]
            dst_rel_trans_weight = [self.rel_trans_weight[etype]
                                    for etype in dst_etypes]

            # Tensor, shape (heads_num * hidden_dim)
            dst_node_rel_fusion_feature = self.relation_fusing(
                dst_node_feat,
                dst_rel_emb,
                dst_node_trans_weight,
                dst_rel_trans_weight
            )

            rel_fusion_emb_dict[dst_type] = dst_node_rel_fusion_feature

        # relation_fusion_embedding_dict, {ntype: tensor -> (nodes, n_heads * hidden_dim)}
        # relation_target_node_features, {(srctype, etype, dst_type): (dst_nodes, n_heads * hidden_dim)}
        return rel_fusion_emb_dict


class RHGNNLayer(nn.Module):
    def __init__(self,
                 ntypes,
                 etypes,
                 in_dim: int,
                 hid_dim: int,
                 rel_in_dim: int,
                 rel_hid_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.2,
                 negative_slope: float = 0.2,
                 use_res: bool = True,
                 use_norm: bool = False):
        """

        :param ntypes: heterogeneous graph's node types
        :param etypes: heterogeneous graph's edge types
        :param in_dim: int, node input dimension
        :param hid_dim: int, node hidden dimension
        :param rel_in_dim: int, relation input dimension
        :param rel_hid_dim: int, relation hidden dimension
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        :param negative_slope: float, negative slope
        :param use_res: boolean, residual connections or not
        :param use_norm: boolean, layer normalization or not
        """
        super(RHGNNLayer, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.rel_in_dim = rel_in_dim
        self.rel_hid_dim = rel_hid_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.use_res = use_res
        self.use_norm = use_norm

        # node transformation parameters of each type
        self.node_trans_weight = nn.ParameterDict({
            ntype: nn.Parameter(torch.randn(in_dim, hid_dim * num_heads))
            for ntype in ntypes
        })

        # relation transformation parameters of each type, used as attention queries
        self.rel_trans_weight = nn.ParameterDict({
            etype: nn.Parameter(torch.randn(rel_in_dim, num_heads * 2 * hid_dim))
            for etype in etypes
        })

        # relation propagation layer of each relation
        self.rel_prop_layer = nn.ModuleDict({
            etype: nn.Linear(rel_in_dim, num_heads * rel_hid_dim)
            for etype in etypes
        })

        # hetero conv modules, each RelationGraphConv deals with a single type of relation
        self.hetero_conv = HeteroGraphConv({
            etype: RelationGraphConv(
                in_dims=(in_dim, in_dim), out_dim=hid_dim, num_heads=num_heads,
                dropout=dropout, negative_slope=negative_slope
            )
            for etype in etypes
        })

        if self.use_res:
            # residual connection
            self.res_fc = nn.ModuleDict()
            self.res_weight = nn.ParameterDict()
            for ntype in ntypes:
                self.res_fc[ntype] = nn.Linear(in_dim, num_heads * hid_dim)
                self.res_weight[ntype] = nn.Parameter(torch.randn(1))

        if self.use_norm:
            self.layer_norm = nn.ModuleDict(
                {ntype: nn.LayerNorm(num_heads * hid_dim) for ntype in ntypes}
            )

        # relation type crossing attention trainable parameters
        self.rel_cross_attn = nn.ParameterDict({
            etype: nn.Parameter(torch.randn(num_heads, hid_dim))
            for etype in etypes
        })
        # different relations crossing layer
        self.rel_cross_layer = RelationCrossing(
            in_feats=num_heads * hid_dim,
            out_feats=hid_dim,
            num_heads=num_heads,
            dropout=dropout,
            negative_slope=negative_slope
        )

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        for weight in self.node_trans_weight:
            nn.init.xavier_normal_(self.node_trans_weight[weight], gain=gain)
        for weight in self.rel_trans_weight:
            nn.init.xavier_normal_(self.rel_trans_weight[weight], gain=gain)
        for etype in self.rel_prop_layer:
            nn.init.xavier_normal_(self.rel_prop_layer[etype].weight, gain=gain)
        if self.use_res:
            for ntype in self.res_fc:
                nn.init.xavier_normal_(self.res_fc[ntype].weight, gain=gain)
        for weight in self.rel_cross_attn:
            nn.init.xavier_normal_(self.rel_cross_attn[weight], gain=gain)

    def forward(self, graph: dgl.DGLHeteroGraph, rel_node_feat: dict, rel_emb: dict):
        """

        :param graph: dgl.DGLHeteroGraph
        :param rel_node_feat: dict, {relation_type: target_node_features shape (N_nodes, input_dim)},
               each value in relation_target_node_features represents the representation of target node features
        :param rel_emb: embedding for each relation, dict, {etype: feature}
        :return: out_feat: dict, {relation_type: target_node_features}
        """
        # in each relation, target type of nodes has an embedding
        # dictionary of {(src_type, etypye, dst_type): target_node_features}
        input_src = rel_node_feat
        input_dst = rel_node_feat

        # out_feat, dict {(src_type, etypye, dst_type): target_node_features}
        out_feat = self.hetero_conv(
            graph, input_src, input_dst, rel_emb,
            self.node_trans_weight, self.rel_trans_weight
        )

        # residual connection for the target node
        if self.use_res:
            for src_type, etype, dst_type in out_feat:
                alpha = F.sigmoid(self.res_weight[dst_type])
                can_edge = (src_type, etype, dst_type)
                res_fc = self.res_fc[dst_type]
                out_feat[can_edge] = (
                        out_feat[can_edge] * alpha +
                        res_fc(input_dst[can_edge]) * (1 - alpha)
                )

        out_feat_dict = {}
        # different relations crossing layer
        for src_type, etype, dst_type in out_feat:
            # (dsttype_node_relations_num, dst_nodes_num, n_heads * hidden_dim)
            can_edge = (src_type, etype, dst_type)
            dst_node_rel_feat = [out_feat[can_edge]
                                 for stype, reltype, dtype in out_feat
                                 if dtype == dst_type]
            dst_node_rel_feat = torch.stack(dst_node_rel_feat, dim=0)

            out_feat_dict[can_edge] = self.rel_cross_layer(
                dst_node_rel_feat,
                self.rel_cross_attn[etype]
            )

        # layer norm for the output
        if self.use_norm:
            for can_edge in out_feat_dict:
                out_feat_dict[can_edge] = self.layer_norm[can_edge[-1]](
                    out_feat_dict[can_edge]
                )

        rel_embd_dict = {}
        for etype in rel_emb:
            rel_embd_dict[etype] = self.rel_prop_layer[etype](rel_emb[etype])

        # relation features after relation crossing layer, {(src_type, etype, dst_type): target_node_features}
        # relation embeddings after relation update, {etype: relation_embedding}
        return out_feat_dict, rel_embd_dict
