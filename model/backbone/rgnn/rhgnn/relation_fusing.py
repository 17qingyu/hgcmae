import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationFusing(nn.Module):

    def __init__(self, node_hid_dim: int, rel_hid_dim: int, num_heads: int, dropout: float = 0.0,
                 negative_slope: float = 0.2):
        """

        :param node_hid_dim: int, node hidden feature size
        :param rel_hid_dim: int,relation hidden feature size
        :param num_heads: int, number of heads in Multi-Head Attention
        :param dropout: float, dropout rate, defaults: 0.0
        :param negative_slope: float, negative slope, defaults: 0.2
        """
        super(RelationFusing, self).__init__()
        self.node_hid_dim = node_hid_dim
        self.rel_hid_dim = rel_hid_dim
        self.num_heads = num_heads

        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self, node_feat: list, rel_emb: list,
                feat_trans_weight: list,
                emb_trans_weight: list):
        """
        :param node_feat: list, [each shape is (num_dst_nodes, n_heads * node_hidden_dim)]
        :param rel_emb: list, [each shape is (n_heads * relation_hidden_dim)]
        :param feat_trans_weight: list, [each shape is (n_heads, node_hidden_dim, node_hidden_dim)]
        :param emb_trans_weight:  list, [each shape is (n_heads, relation_hidden_dim, relation_hidden_dim)]
        :return: rel_fusion_feat: Tensor of the target node representation after relation-aware representations fusion
        """
        if len(node_feat) == 1:
            # (num_dst_nodes, n_heads * hidden_dim)
            rel_fusion_feat = node_feat[0]
        else:
            # (num_dst_relations, nodes, n_heads, node_hidden_dim)
            node_feat = torch.stack(node_feat, dim=0).reshape(len(node_feat), -1,
                                                              self.num_heads, self.node_hid_dim)
            # (num_dst_relations, n_heads, relation_hidden_dim)
            rel_emb = torch.stack(rel_emb, dim=0).reshape(len(node_feat),
                                                          self.num_heads,
                                                          self.rel_hid_dim)
            # (num_dst_relations, n_heads, node_hidden_dim, node_hidden_dim)
            feat_trans_weight = torch.stack(feat_trans_weight, dim=0).reshape(
                len(node_feat), self.num_heads,
                self.node_hid_dim, self.node_hid_dim)
            # (num_dst_relations, n_heads, relation_hidden_dim, relation_hidden_dim)
            emb_trans_weight = torch.stack(emb_trans_weight,
                                           dim=0).reshape(len(node_feat),
                                                          self.num_heads,
                                                          self.rel_hid_dim,
                                                          self.node_hid_dim)
            # shape (num_dst_relations, nodes, n_heads, hidden_dim)
            node_feat = torch.einsum('abcd,acde->abce', node_feat,
                                     feat_trans_weight)

            # shape (num_dst_relations, n_heads, hidden_dim)
            rel_emb = torch.einsum('abc,abcd->abd', rel_emb,
                                   emb_trans_weight)

            # shape (num_dst_relations, nodes, n_heads, 1)
            atten_scores = ((node_feat * rel_emb.unsqueeze(dim=1))
                            .sum(dim=-1, keepdim=True))
            atten_scores = F.softmax(self.leaky_relu(atten_scores), dim=0)
            # (nodes, n_heads, hidden_dim)
            rel_fusion_feat = (node_feat * atten_scores).sum(dim=0)
            rel_fusion_feat = self.dropout(rel_fusion_feat)
            # (nodes, n_heads * hidden_dim)
            rel_fusion_feat = rel_fusion_feat.reshape(
                -1, self.num_heads * self.node_hid_dim
            )

        return rel_fusion_feat
