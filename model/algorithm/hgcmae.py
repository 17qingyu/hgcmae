from lightning import LightningModule
import torch
import torch.nn as nn


from .evaluation import *


class HGCMAE(LightningModule):
    def __init__(
        self,
        mask: nn.Module,
        backbone: dict[str, nn.Module],
        neck: dict[str, nn.Module],
        head: nn.Module,
        base_momentum: float = 0.996,
        non_target_weight: float = 0.5
    ):
        super(HGCMAE, self).__init__()
        self.backbone = backbone["online"]
        self.target_backbone = backbone["target"]
        self.mask = mask
        self.feature_decoder = neck["decoder"]
        self.project = neck["project"]
        self.target_project = neck["target_project"]
        self.non_target_weight = non_target_weight

        assert head is not None

        self.head = head
        self.momentum = base_momentum
        self.init_parameter()

    def init_parameter(self):
        for param_b, param_m in zip(self.backbone.parameters(),
                                    self.target_backbone.parameters()):
            param_m.data.copy_(param_b.data)
            param_m.requires_grad = False

        for param_b, param_m in zip(self.project.parameters(),
                                    self.target_project.parameters()):
            param_m.data.copy_(param_b.data)
            param_m.requires_grad = False

    @torch.no_grad
    def momentum_update(self):
        for param_b, param_m in zip(self.backbone.parameters(),
                                    self.target_backbone.parameters()):
            param_m.data = param_b.data * \
                (1 - self.momentum) + param_m.data * self.momentum

        for param_b, param_m in zip(self.project.parameters(),
                                    self.target_project.parameters()):
            param_m.data = param_b.data * \
                (1 - self.momentum) + param_m.data * self.momentum

    def forward(self, graph, mask_x_s, mask_x_t, target_type, mask_nodes, pos):
        embed_s = self.backbone(graph, mask_x_s)
        embed_t = self.target_backbone(graph, mask_x_t)
        restor_feat = self.feature_decoder(graph, embed_s)

        pro_embed_s = self.project(embed_s[target_type])
        pro_embed_t = self.target_project(embed_t[target_type])

        loss = self.head(graph, pro_embed_s, pro_embed_t, mask_x_t,
                         restor_feat, mask_nodes,target_type, pos)
        train_loss = loss["rc_loss"] + loss["ct_loss"]

        return train_loss

    def forward_test(self, graph):
        embed_online = self.backbone(graph, graph.ndata["x"])
        return embed_online

    def training_step(self, batch, batch_idx):
        graph, target_type, pos = batch
        mask_x_s, mask_x_t, mask_s, mask_t = self.mask(
            graph.ndata["x"], self.current_epoch, target_type)

        loss_target = self(graph, mask_x_s, mask_x_t, target_type, mask_s, pos)
        loss_non = self(graph, mask_x_t, mask_x_s, target_type, mask_t, pos)
        self.log("loss_target", loss_target.item())
        self.log("loss_non", loss_non.item())
        loss = loss_target + self.non_target_weight * loss_non
        self.log("train_loss", loss.item())
        return loss

    def _evaluation(self, graph, target_type):
        embed = self.forward_test(graph)
        label = graph.nodes[target_type].data["y"]
        embed = embed[target_type].cpu().detach()
        label = label.cpu().detach()

        save_dir = self.trainer.logger.log_dir if self.trainer.logger else "data"

        distances, sim_matrix = evaluate_clustering(embed, label)
        visualize_embeddings(embed, label, save_dir, self.current_epoch)
        print(f"distance of class: {distances}")
        print(f"Class Center Similarity Matrix:\n{sim_matrix}")

    def validation_step(self, batch, batch_idx):
        self._evaluation(*batch)

    def test_step(self, batch, batch_idx):
        self._evaluation(*batch)
