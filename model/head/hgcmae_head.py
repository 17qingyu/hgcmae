import dgl
import torch
import torch.nn as nn


class HGCMAEPretrainHead(nn.Module):
    def __init__(self, ct_loss, rc_loss, ct_weight, rc_weight):
        super().__init__()
        self.ct_loss = ct_loss
        self.rc_loss = rc_loss
        self.ct_weight = ct_weight
        self.rc_weight = rc_weight
        
    def forward(self, graph, pro_embed_s, pro_embed_t, feat, restor_feat, mask_nodes, target_type, pos):
        rc_loss_all = 0
        for ntyp, mask_node in mask_nodes.items():
            if mask_node.shape[0]:
                rc_loss = self.rc_loss(
                    feat[ntyp][mask_node], restor_feat[ntyp][mask_node])
                rc_loss_all += rc_loss

        ct_loss = self.ct_loss(pro_embed_s, pro_embed_t, pos)

        return {"ct_loss": self.ct_weight * ct_loss, "rc_loss": self.rc_weight * rc_loss_all}
    