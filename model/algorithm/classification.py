from lightning.pytorch import LightningModule
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

class Classification(LightningModule):
    def __init__(self, pretrained_model: nn.Module, in_dim, out_dim):
        super().__init__()
        self.feature_extractor = pretrained_model
        self.classifier = torch.nn.Linear(in_dim, out_dim)
        self.embed = None
        self.label = None
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize classifier weights"""
        # Xavier/Glorot initialization for linear layer
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def on_fit_start(self):
        if self.trainer.datamodule is not None:
            datamodule = self.trainer.datamodule
            self.target_type = datamodule.dataset.target_type

    def feature_extract(self, g, x):
        if self.embed is None:
            self.feature_extractor.eval()
            embed_dict = self.feature_extractor(g, x)
            self.embed = embed_dict[self.target_type].detach()
            # self.embed = torch.load(
            #     "/root/siton-data-houyusen/gyl/hgcmae/logs/pretrain/freebase/version_213/embeddings_999", self.device)
            # self.embed = torch.load(
            #     "/root/siton-data-houyusen/gyl/hgcmae/logs/pretrain/aminer/version_100/embeddings_799", self.device)
            # self.embed = torch.load(
            #     "/root/siton-data-houyusen/gyl/hgcmae/logs/pretrain/aminer/version_45/embeddings_1999", self.device)
            self.label = g.nodes[self.target_type].data["y"]

    def forward(self, g, x):
        self.feature_extract(g, x)
        return self.classifier(self.embed)

    def training_step(self, g, batch_id):
        predict = self(g, g.ndata["x"])
        train_mask = g.nodes[self.target_type].data["train_mask"]
        loss = F.cross_entropy(
            predict[train_mask], self.label[train_mask]).mean()
        self.log("train_loss", loss.item())
        return loss

    def validation_step(self, g, batch_id):
        predict = self(g, g.ndata["x"])
        val_mask = g.nodes[self.target_type].data["val_mask"]
        loss = F.cross_entropy(predict[val_mask], self.label[val_mask]).mean()
        self.log("val_loss", loss.item(), batch_size=1)

    # def test_step(self, g, batch_id):
    #     predict = self(g, g.ndata["x"])
    #     mask = g.nodes[self.target_type].data["test_mask"]
    #     loss = F.cross_entropy(predict[mask], self.label[mask]).mean()
    #     pred_label = torch.argmax(predict, dim=1)
    #     f1_macro, f1_micro = self.get_f1_score(
    #         pred_label[mask], self.label[mask])
    #     self.log("test_loss", loss.item(), batch_size=1)
    #     self.log("f1_macro", f1_macro, batch_size=1)
    #     self.log("f1_micro", f1_micro, batch_size=1)
    #     print(f"test loss({loss}) f1_macro({f1_macro}) f1_micro({f1_micro})")

    def test_step(self, g, batch_id):
        predict = self(g, g.ndata["x"])
        mask = g.nodes[self.target_type].data["test_mask"]
        loss = F.cross_entropy(predict[mask], self.label[mask]).mean()
        
        # softmax 得到每一类的概率
        prob = F.softmax(predict[mask], dim=1).detach().cpu().numpy()
        true_label = self.label[mask].detach().cpu().numpy()
        
        pred_label = torch.argmax(predict, dim=1)
        f1_macro, f1_micro = self.get_f1_score(
            pred_label[mask], self.label[mask])

        # 处理为 one-hot 标签
        num_classes = predict.shape[1]
        try:
            y_true_bin = label_binarize(true_label, classes=range(num_classes))
            auc = roc_auc_score(y_true_bin, prob, average="macro", multi_class="ovr")
        except ValueError:
            auc = 0.0  # 可能只有一个类，导致报错

        self.log("test_loss", loss.item(), batch_size=1)
        self.log("f1_micro", f1_micro, batch_size=1)
        self.log("f1_macro", f1_macro, batch_size=1)
        self.log("auc", auc, batch_size=1)

        print(f"test loss({loss}) f1_micro({f1_micro}) f1_macro({f1_macro}) auc({auc})")

    def get_f1_score(self, pred_label, true_label):
        f1_macro = f1_score(
            pred_label.cpu(), true_label.cpu(), average='macro')
        f1_micro = f1_score(
            pred_label.cpu(), true_label.cpu(), average='micro')
        return f1_macro, f1_micro

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.008, weight_decay=1e-4)
