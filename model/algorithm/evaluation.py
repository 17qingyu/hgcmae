import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import torch
import torch.nn.functional as F


def evaluate_clustering(features: torch.Tensor, labels: torch.Tensor):
    unique_labels = np.unique(labels)

    class_centers = []
    distances = []
    for l in unique_labels:
        class_features = features[labels == l]
        class_centers.append(class_features.mean(dim=0))
        cos_sim = F.cosine_similarity(
            class_features, class_centers[-1].unsqueeze(0))
        distances.append((1 - cos_sim).mean())
    class_centers = torch.stack(class_centers)

    sim_matrix = torch.nn.functional.cosine_similarity(
        class_centers.unsqueeze(1),
        class_centers.unsqueeze(0),
        dim=-1
    )

    return distances, sim_matrix


def visualize_embeddings(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    save_dir: str,
    epoch: int
):
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1], s=3, c=labels, cmap="jet", alpha=0.7)
    plt.colorbar(scatter)
    plt.title("Node Embeddings Visualization")
    plt.xlabel("TSNE-1")
    plt.ylabel("TSNE-2")

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"embeddings_{epoch}.png"))
    plt.close()

    torch.save(embeddings, os.path.join(save_dir, f"embeddings_{epoch}"))
