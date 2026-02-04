import torch
import dgl
from collections import defaultdict
from dataset import ACMDataset, AminerDataset, DBLPDataset, FreebaseDataset, IMDBDataset

@torch.no_grad
def _generate_adj_sparse(g: dgl.DGLHeteroGraph, device="cpu"):
    ntypes = g.ntypes
    num_nodes = g.num_nodes()
    num_nodes_dict = {ntyp: g.num_nodes(ntype=ntyp) for ntyp in ntypes}

    current = 0
    re_index = {}
    for ntyp, num_node in num_nodes_dict.items():
        re_index[ntyp] = (current, current + num_node - 1)
        current += num_node

    edge_list_all = []
    rel_adj = defaultdict(list)

    for src_type, etype, dst_type in g.canonical_etypes:
        src, dst = g.edges(etype=etype)
        src = src + re_index[src_type][0]
        dst = dst + re_index[dst_type][0]
        edge_index = torch.stack([src, dst])

        edge_list_all.append(edge_index)
        rel_adj[src_type].append(edge_index)
        rel_adj[dst_type].append(edge_index)

    # 全图邻接矩阵
    edge_index_all = torch.cat(edge_list_all, dim=1)
    values_all = torch.ones(edge_index_all.shape[1], device=device)
    adj = torch.sparse_coo_tensor(
        edge_index_all, values_all, (num_nodes, num_nodes)).coalesce()

    # 每种节点类型的子图邻接矩阵
    rel_adj_sparse = {}
    for ntype, edge_list in rel_adj.items():
        edges = torch.cat(edge_list, dim=1)
        values = torch.ones(edges.shape[1], device=device)
        rel_adj_sparse[ntype] = torch.sparse_coo_tensor(
            edges, values, (num_nodes, num_nodes)).coalesce()

    return re_index, adj, rel_adj_sparse


@torch.no_grad
def _generate_pos_sparse(g: dgl.DGLHeteroGraph, k: int, target_type: str, device: str = "cpu", batch_size=512, forbidden_rel=None):
    if forbidden_rel is None:
        forbidden_rel = []

    re_index, adj, rel_adj = _generate_adj_sparse(g, device=device)

    target_start, target_end = re_index[target_type]
    num_target = target_end - target_start + 1

    total_pos = torch.sparse_coo_tensor(
        torch.tensor([[], []]),
        torch.tensor([]),
        (num_target, num_target),
        device=device
    )
    for ntype in g.ntypes:
        if ntype == target_type or ntype in forbidden_rel:
            continue

        forbidden_adj = adj - rel_adj[ntype]
        khop_rel_adj = forbidden_adj.clone()
        all_adj = adj.clone()

        for _ in range(k - 1):
            khop_rel_adj = torch.sparse.mm(khop_rel_adj, forbidden_adj)
            all_adj = torch.sparse.mm(all_adj, adj)
            khop_rel_adj = khop_rel_adj.coalesce()
            all_adj = all_adj.coalesce()

        # 正样本 = 全图传播 - 禁止关系传播
        positive_adj = all_adj - khop_rel_adj
        positive_adj = positive_adj.coalesce()

        del all_adj, khop_rel_adj

        # 只取目标节点之间的部分
        rows = positive_adj.indices()[0]
        cols = positive_adj.indices()[1]
        values = positive_adj.values()

        mask = (rows >= target_start) & (rows <= target_end) & \
               (cols >= target_start) & (cols <= target_end)

        filtered_rows = rows[mask] - target_start
        filtered_cols = cols[mask] - target_start
        filtered_values = values[mask]

        sparse_tensor = torch.sparse_coo_tensor(
            torch.stack([filtered_rows, filtered_cols]),
            filtered_values,
            (num_target, num_target),
            device=device
        ).coalesce()

        row_sum = torch.sparse.sum(
            sparse_tensor, dim=1).to_dense().unsqueeze(-1)
        row_sum[row_sum == 0] = 1  # 防止除以零

        norm_values = sparse_tensor.values(
        ) / row_sum[sparse_tensor.indices()[0], 0]
        sparse_tensor = torch.sparse_coo_tensor(
            sparse_tensor.indices(), norm_values, sparse_tensor.shape).coalesce()

        total_pos = total_pos + sparse_tensor

    return total_pos.coalesce()


@torch.no_grad
def generate_pos(graph, k, target_type, device, pos_num, save_path=".", forbidden_rel: list = None):
    weight = _generate_pos_sparse(
        graph, k, target_type, device, forbidden_rel=forbidden_rel)

    num_target = graph.num_nodes(target_type)
    n_positive = torch.sum(weight.to_dense() > 0, dim=1).to(torch.float32)
    print("Positive edge weights:", n_positive.min(),
          n_positive.mean(), n_positive.max())

    positive_sample = torch.zeros((num_target, num_target), device=device)

    indices = weight.indices()
    values = weight.values()
    row, col = indices[0], indices[1]

    nonzero_mask = values > 0
    values = values[nonzero_mask]
    row = row[nonzero_mask]
    col = col[nonzero_mask]
    
    for i in range(num_target):
        mask = (row == i)
        candidates = col[mask]
        weights = values[mask]

        if candidates.numel() > pos_num:
            topk = torch.topk(weights, pos_num)
            selected = candidates[topk.indices]
        else:
            selected = candidates

        positive_sample[i, selected] = 1

    label = graph.nodes[target_type].data["y"]
    indices = positive_sample.nonzero().T
    u, v = indices[0], indices[1]
    num_same_edge = (label[u] == label[v]).sum()
    print(
        f"len of egde({positive_sample.sum()}) len of same label({num_same_edge}) rate({num_same_edge / positive_sample.sum()})")

    torch.save(positive_sample.cpu(), save_path)

def load_dataset(dataset_name):
    if dataset_name == "acm":
        return ACMDataset("data")
    elif dataset_name == "aminer":
        return AminerDataset("data")
    elif dataset_name == "freebase":
        return FreebaseDataset("data")
    elif dataset_name == "dblp":
        return DBLPDataset("data")
    elif dataset_name == "imdb":
        return IMDBDataset("data")
    else:
        print(f"unknown dataset {dataset_name}")
        exit(-1)


if __name__ == "__main__":
    dataset_name = "aminer"
    
    file_path = f"/root/siton-data-houyusen/gyl/hgcmae/data/{dataset_name}/processed/{dataset_name}.dgl"


    target_type_dict = {
        "acm": "paper",
        "dblp": "author",
        "freebase": "movie",
        "aminer": "paper",
        "imdb": "movie"
    }

    k_dict = {
        "acm": 2,
        "dblp": 4,
        "freebase": 2,
        "aminer": 2,
        "imdb": 2
    }

    pos_num_dict = {
        "acm": 7,
        "dblp": 1000,
        "freebase": 70,
        "aminer": 25,
        "imdb": 5
    }
    device = "cuda:0"

    save_path = f"/root/siton-data-houyusen/gyl/hgcmae/data/{dataset_name}/processed/pos.th"
    dataset = load_dataset(dataset_name)
    graph = dataset[0].to(device)


    import time

    start = time.time()

    generate_pos(graph, k_dict[dataset_name], target_type_dict[dataset_name],
                 device, pos_num_dict[dataset_name], save_path)

    end = time.time()
    print(f"运行时间: {end - start:.6f} 秒")
