import torch
import torch.nn.functional as F


class CosSimLoss:
    def __init__(self, alpha=1):
        self.alpha = alpha
    
    def __call__(self, pred, target, *args):
        normal_pred = F.normalize(pred, dim=-1)
        normal_target = F.normalize(target, dim=-1)
        cosine_similarity = (normal_pred * normal_target).sum(dim=-1).pow_(self.alpha)
        loss = 1 - cosine_similarity
        return loss.mean()


class InfoNceLoss:
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def __call__(self, query: torch.Tensor, key: torch.Tensor, labels: torch.Tensor = None):
        """
        计算 InfoNCE 损失

        参数:
            query: torch.Tensor, shape (N, D) - 查询向量
            key: torch.Tensor, shape (N, D) - 键向量
            labels: torch.Tensor, shape (N, N) - 0-1 标签矩阵，labels[i][j]=1 表示 query[i] 和 key[j] 是正样本对

        返回:
            loss: torch.Tensor, InfoNCE 损失标量
        """
        # 归一化特征向量
        query = F.normalize(query, p=2, dim=1)  # (N, D)
        key = F.normalize(key, p=2, dim=1)      # (N, D)

        # 计算相似度矩阵 (N, N)
        similarity_matrix = torch.matmul(query, key.T) / self.temperature  # (N, N)
        # similarity_matrix = torch.relu(similarity_matrix)

        if labels is None:
            # 默认情况下，假设对角线为正样本对
            labels = torch.eye(similarity_matrix.size(0), device=similarity_matrix.device)

        # 将标签转换为概率分布
        labels = labels / labels.sum(dim=1, keepdim=True)

        # 计算交叉熵损失
        loss = - (labels * F.log_softmax(similarity_matrix, dim=1)).sum(dim=1).mean()

        return loss

