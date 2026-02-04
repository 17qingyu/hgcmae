from collections import defaultdict

import dgl
import numpy as np
import torch


@torch.no_grad
def _generate_adj(g: dgl.DGLHeteroGraph, device="cpu") -> tuple[dict[str, tuple], torch.Tensor, torch.Tensor]:
    ntypes = g.ntypes
    num_nodes = g.num_nodes()
    num_nodes_dict = {
        ntyp: g.num_nodes(ntype=ntyp)
        for ntyp in ntypes
    }

    current = 0
    re_index = {}
    for ntyp, num_node in num_nodes_dict.items():
        re_index[ntyp] = (current, current + num_node - 1)
        current += num_node

    adj = torch.zeros((num_nodes, num_nodes), device=device)

    rel_adj = defaultdict(lambda: torch.zeros(
        (num_nodes, num_nodes), device=device))
    for src_type, etype, dst_type in g.canonical_etypes:
        edges = g.edges(etype=etype)
        re_src_index = edges[0] + re_index[src_type][0]
        re_dst_index = edges[1] + re_index[dst_type][0]
        sub_adj = torch.zeros((num_nodes, num_nodes), device=device)
        sub_adj[re_src_index, re_dst_index] = 1

        rel_adj[dst_type] += sub_adj
        rel_adj[src_type] += sub_adj
        adj += rel_adj[dst_type]

    return re_index, adj, rel_adj


@torch.no_grad
def _generate_pos(g: dgl.DGLHeteroGraph, k: int, target_type: str, device: str = "cpu"):
    re_index, adj, rel_adj = _generate_adj(g, device=device)

    row = torch.arange(re_index[target_type][0],
                       re_index[target_type][1] + 1, device=device)
    col = torch.arange(re_index[target_type][0],
                       re_index[target_type][1] + 1, device=device)
    row, col = torch.meshgrid(row, col, indexing='ij')

    postive_sample = []
    for ntype in g.ntypes:
        if ntype == target_type:
            continue

        forbiden_rel_adj = adj - rel_adj[ntype]
        khop_rel_adj = adj - rel_adj[ntype]
        all_adj = adj.clone()

        for i in range(k - 1):
            khop_rel_adj = khop_rel_adj @ forbiden_rel_adj
            all_adj = all_adj @ adj

            positive_adj = all_adj - khop_rel_adj

            target_adj = positive_adj[row, col]
            if target_adj.sum() > 0:
                target_adj = target_adj / target_adj.sum(dim=1)
                postive_sample.append(target_adj)

    postive_sample = torch.stack(postive_sample)
    postive_sample = torch.sum(postive_sample, dim=0)

    return postive_sample


@torch.no_grad
def generate_pos(graph, k, target_type, device, pos_num, save_path="."):
    weight = _generate_pos(graph, k, target_type, device)

    n_positive = (weight > 0).sum(-1).to(torch.float32)
    print(n_positive.max(), n_positive.min(), n_positive.mean())

    positive_sample = torch.zeros(
        (graph.num_nodes(target_type), graph.num_nodes(target_type)))

    for i in range(weight.shape[0]):
        nonzero_weight = weight[i].nonzero().flatten()
        if nonzero_weight.shape[0] > pos_num:
            oo = torch.argsort(weight[i, nonzero_weight], descending=True)
            sele = nonzero_weight[oo[:pos_num]]
            positive_sample[i, sele] = 1
        else:
            positive_sample[i, nonzero_weight] = 1
            
    label = graph.nodes[target_type].data["y"]
    indices = positive_sample.nonzero().T
    u, v = indices[0], indices[1]
    num_same_edge = (label[u] == label[v]).sum()
    print(
        f"len of egde({positive_sample.sum()}) len of same label({num_same_edge}) rate({num_same_edge / positive_sample.sum()})")

    torch.save(positive_sample, save_path)


if __name__ == "__main__":
    dataset = "dblp"
    file_path = f"/root/siton-data-houyusen/gyl/hgcmae/data/dblp/processed/dblp.dgl"
    save_path = f"/root/siton-data-houyusen/gyl/hgcmae/data/dblp/processed/pos.th"

    target_type_dict = {
        "acm": "paper",
        "dblp": "author",
        "freebase": "movie",
        "aminer": "paper"
    }

    k_dict = {
        "acm": 2,
        "dblp": 4,
        "freebase": 2,
        "aminer": 2
    }

    pos_num_dict = {
        "acm": 7,
        "dblp": 1000,
        "freebase": 20,
        "aminer": 15
    }
    device = "cuda:0"

    graphs, *_ = dgl.load_graphs(file_path)
    graph = graphs[0].to(device)

    generate_pos(graph, k_dict[dataset], target_type_dict[dataset],
                 device, pos_num_dict[dataset], save_path)
