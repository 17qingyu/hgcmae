import torch.nn as nn
import dgl


class HeteroGraphConv(nn.Module):
    r"""A generic module for computing convolution on heterogeneous graphs.

    The heterograph convolution applies sub-modules on their associating
    relation graphs, which reads the features from source nodes and writes the
    updated ones to destination nodes. If multiple relations have the same
    destination node types, their results are aggregated by the specified method.

    If the relation graph has no edge, the corresponding module will not be called.

    Parameters
    ----------
    mods : dict[str, nn.Module]
        Modules associated with every edge types.
    """

    def __init__(self, mods: dict[str, nn.Module]):
        super(HeteroGraphConv, self).__init__()
        self.mods = nn.ModuleDict(mods)

    def forward(
            self,
            graph: dgl.DGLHeteroGraph,
            input_src: dict,
            input_dst: dict,
            rel_emb: dict,
            node_trans_weight: nn.ParameterDict,
            rel_trans_weight: nn.ParameterDict
    ):
        """
        call the forward function with each module.

        Parameters
        ----------
        graph: DGLHeteroGraph, The Heterogeneous Graph.
        input_src: dict[tuple, Tensor], Input source node features {relation_type: features, }
        input_dst: dict[tuple, Tensor], Input destination node features {relation_type: features, }
        rel_emb: dict[etype, Tensor], Input relation features {etype: feature}
        node_trans_weight: nn.ParameterDict, weights {ntype, (inp_dim, hidden_dim)}
        rel_trans_weight: nn.ParameterDict, weights {etype, (n_heads, 2 * hidden_dim)}

        Returns
        -------
        outputs, dict[tuple, Tensor]  Output representations for every relation -> {(stype, etype, dtype): features}.
        """

        # find reverse relation dict
        can_etypes = list(input_src.keys())
        reverse_rel_dict = {}
        for i in range(len(input_src)):
            for j in range(i + 1, len(input_src)):
                etype1 = can_etypes[i]
                etype2 = can_etypes[j]
                if etype1[-1] == etype2[0] and etype1[0] == etype2[-1] and etype1[1] != etype2[1]:
                    reverse_rel_dict[etype1[1]] = etype2[1]
                    reverse_rel_dict[etype2[1]] = etype1[1]
                    break

        # dictionary, {(src_type, etype, dst_type): representations}
        outputs = dict()
        for stype, etype, dtype in graph.canonical_etypes:
            rel_graph = graph[stype, etype, dtype]
            edge = (stype, etype, dtype)
            reverse_edge = (dtype, reverse_rel_dict[etype], stype)
            if rel_graph.number_of_edges() == 0:
                continue
            # for example, (author, writes, paper) relation, take author as src_nodes, take paper as dst_nodes
            dst_repres = self.mods[etype](
                rel_graph,
                (input_src[reverse_edge], input_dst[edge]),
                node_trans_weight[dtype],
                node_trans_weight[stype],
                rel_emb[etype],
                rel_trans_weight[etype]
            )

            # dst_repres (dst_nodes, hid_dim)
            outputs[edge] = dst_repres

        return outputs
