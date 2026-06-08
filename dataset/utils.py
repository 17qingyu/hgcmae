from typing import Mapping, Tuple
import dgl
import torch


class RelBlockMatrix:
    def __init__(
        self,
        sub_adjs: Mapping[Tuple[str, str, str], torch.Tensor],
        device: torch.device | str | None = None,
    ):
        if device is None:
            device = next(iter(sub_adjs.values())).device
        else:
            device = torch.device(device)

        self.device = device
        
        self.sub_adjs = {k: v.to(device) for k, v in sub_adjs.items()}


    @classmethod
    def from_hetero_graph(
        cls,
        g: dgl.DGLHeteroGraph,
        device: torch.device | str = "cpu",
        dtype=torch.float32,
    ):
        device = torch.device(device)
        data = dict()

        for etype in g.canonical_etypes:
            src, dst = g.edges(etype=etype)

            src = src.to(device)
            dst = dst.to(device)

            num_src = g.num_nodes(etype[0])
            num_dst = g.num_nodes(etype[2])

            adj = torch.zeros(
                (num_src, num_dst),
                device=device,
                dtype=dtype,
            )
            adj[src, dst] = 1

            data[etype] = adj

        return cls(data, device=device)

    def to(self, device):
        device = torch.device(device)
        if device == self.device:
            return self
        return RelBlockMatrix(self.sub_adjs, device=device)

    @classmethod
    def remove_ntype(cls, matrix, ntype: str):
        data = {
            k: v for k, v in matrix.sub_adjs.items() if k[0] != ntype and k[-1] != ntype
        }
        return cls(data, device=matrix.device)

    def __getitem__(self, key):
        if key not in self.sub_adjs:
            raise KeyError(f"missing etype {key}")
        return self.sub_adjs[key]

    def __matmul__(self, other: "RelBlockMatrix"):
        if self.device != other.device:
            raise RuntimeError(f"Device mismatch: {self.device} vs {other.device}")

        new_adjs = dict()

        for etype_u, A in self.sub_adjs.items():
            for etype_v, B in other.sub_adjs.items():
                if etype_u[-1] != etype_v[0]:
                    continue

                rel = (
                    etype_u[0],
                    f"{etype_u[0]}-{etype_v[-1]}",
                    etype_v[-1],
                )

                if rel not in new_adjs:
                    new_adjs[rel] = torch.zeros(
                        (A.shape[0], B.shape[1]),
                        device=self.device,
                        dtype=A.dtype,
                    )

                new_adjs[rel] += A @ B

        return RelBlockMatrix(new_adjs, device=self.device)


@torch.no_grad
def _generate_pos(g: dgl.DGLHeteroGraph, k: int, target_type: str, device: str = "cpu"):
    rel_adj = RelBlockMatrix.from_hetero_graph(g, device=device)

    postive_sample = []
    target_etype = (target_type, f"{target_type}-{target_type}", target_type)
    num_target_node = g.num_nodes(target_type)

    for ntype in g.ntypes:
        if ntype == target_type:
            continue

        forbiden_rel_adj = RelBlockMatrix.remove_ntype(rel_adj, ntype)
        k_hop_forbiden_adj = forbiden_rel_adj
        k_hop_adj = rel_adj

        for _ in range(k - 1):
            k_hop_forbiden_adj = k_hop_forbiden_adj @ forbiden_rel_adj
            k_hop_adj = k_hop_adj @ rel_adj

            try:
                temp_forbi_adj = k_hop_forbiden_adj[target_etype]
            except KeyError:
                temp_forbi_adj = torch.zeros((num_target_node, num_target_node), device=device)

            try:
                temp_adj = k_hop_adj[target_etype]
            except KeyError:
                temp_adj = torch.zeros((num_target_node, num_target_node), device=device)

            target_adj = temp_adj - temp_forbi_adj

            if target_adj.sum() > 0:
                target_adj = target_adj / target_adj.sum(dim=1)
                postive_sample.append(target_adj)

    postive_sample = torch.stack(postive_sample)
    postive_sample = torch.sum(postive_sample, dim=0)

    return postive_sample


@torch.no_grad
def generate_pos(graph, k, target_type, device, pos_num):
    weight = _generate_pos(graph, k, target_type, device)

    num_target_node = graph.num_nodes(target_type)

    positive_sample = torch.zeros((num_target_node, num_target_node))

    for i in range(weight.shape[0]):
        nonzero_weight = weight[i].nonzero().flatten()
        if nonzero_weight.shape[0] > pos_num:
            oo = torch.argsort(weight[i, nonzero_weight], descending=True)
            sele = nonzero_weight[oo[:pos_num]]
            positive_sample[i, sele] = 1
        else:
            positive_sample[i, nonzero_weight] = 1

    return positive_sample
