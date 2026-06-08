import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

class Cluster(LightningModule):
    def __init__(self, pretrained_model: nn.Module, in_dim, n_clusters: int, target_type, kmeans_random_state: int = 0):
        super().__init__()
        self.feature_extractor = pretrained_model
        self.in_dim = in_dim
        self.n_clusters = n_clusters
        self.kmeans_random_state = kmeans_random_state
        self.target_type = target_type

        self.embed = None
        self.label = None

    def on_fit_start(self):
        if self.trainer.datamodule is not None:
            datamodule = self.trainer.datamodule
            self.target_type = datamodule.dataset.target_type

    def feature_extract(self, g, x):
        if self.embed is None:
            self.feature_extractor.eval()
            with torch.no_grad():
                embed_dict = self.feature_extractor(g, x)
                self.embed = embed_dict[self.target_type].detach()
                self.label = g.nodes[self.target_type].data["y"]

    def forward(self, g, x):
        self.feature_extract(g, x)
        return self.embed

    def test_step(self, g, batch_id):
        embeds = self(g, g.ndata["x"])  # forward 提取嵌入
        embeds_np = embeds.detach().cpu().numpy()
        labels_np = self.label.detach().cpu().numpy()

        y_pred = KMeans(n_clusters=self.n_clusters, random_state=self.kmeans_random_state).fit_predict(embeds_np)
        nmi = normalized_mutual_info_score(labels_np, y_pred)
        ari = adjusted_rand_score(labels_np, y_pred)

        self.log("nmi", nmi, batch_size=1)
        self.log("ari", ari, batch_size=1)

        print(f"[Cluster Test] NMI: {nmi:.4f}, ARI: {ari:.4f}")