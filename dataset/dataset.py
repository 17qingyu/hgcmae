from itertools import product
import os
import os.path as osp


import dgl
from dgl.data import DGLDataset
from dgl.data.utils import save_graphs, load_graphs
import numpy as np
import scipy.sparse as sp
import torch


class HGMAEDataset(DGLDataset):
    node_types = []

    num_nodes_dict = {}

    target_type = ""

    def __init__(self,
                 name,
                 raw_dir=None,
                 force_reload=False,
                 verbose=False,
                 transform=None):
        super().__init__(name=name,
                         raw_dir=raw_dir,
                         force_reload=force_reload,
                         verbose=verbose,
                         transform=transform)

    @property
    def root(self):
        return os.path.join(self.raw_dir, self.name)

    @property
    def raw_path(self):
        return os.path.join(self.raw_dir, self.name, "raw_dir")

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def save_name(self):
        return f"{self.name}.dgl"

    @property
    def save_dir(self):
        return os.path.join(self.raw_dir, self.name, "processed")

    @property
    def graph_path(self):
        return os.path.join(self.save_dir, self.save_name)

    def has_cache(self):
        return osp.exists(self.graph_path)

    def download(self):
        pass

    def process(self):
        edge_index_dict = {}
        for src, dst in product(self.node_types, self.node_types):
            file_name = osp.join(self.raw_path, f'{src}-{dst}.txt')

            if not osp.exists(file_name):
                continue

            with open(file_name, "r") as f:
                edges = f.readlines()
            edges = [(int(u), int(v))
                     for u, v in (edge.split() for edge in edges)]

            etype = (src, f"{src}-{dst}", dst)
            edge = torch.tensor(edges, dtype=torch.long).T
            edge_index_dict[etype] = (edge[0], edge[1])

            revers_etype = (dst, f"{dst}-{src}", src)
            edge_index_dict[revers_etype] = (edge[1], edge[0])

        g = dgl.heterograph(
            edge_index_dict, num_nodes_dict=self.num_nodes_dict)

        for node_type in self.node_types:
            file_name = osp.join(self.raw_path, f'feature_{node_type}.npz')
            if os.path.exists(file_name):
                x = sp.load_npz(file_name)
                g.nodes[node_type].data['x'] = torch.from_numpy(
                    x.todense()).to(torch.float)
            else:
                g.nodes[node_type].data['x'] = torch.eye(
                    self.num_nodes_dict[node_type]).to(torch.float)

        y = np.load(osp.join(self.raw_path, 'labels.npy'))
        g.nodes[self.target_type].data['y'] = torch.from_numpy(
            y).to(torch.long)

        num_target_node = self.num_nodes_dict[self.target_type]
        for rate in [20, 40, 60]:
            for name in ['train', 'val', 'test']:
                if osp.exists(osp.join(self.raw_path, f'{name}_{rate}.npy')):
                    idx = np.load(osp.join(self.raw_path, f'{name}_{rate}.npy'))
                    idx = torch.from_numpy(idx).to(torch.long)
                    mask = torch.zeros(num_target_node, dtype=torch.bool)
                    mask[idx] = True
                    g.nodes[self.target_type].data[f'{name}_{rate}_mask'] = mask

        self._num_classes = np.unique(y).size
        self._g = g

        if self.verbose:
            self._print_graph_info()

    def _print_graph_info(self):
        print(f"dataset {self.name}'s info: ")

        print(f"\tNumNodes: ", end="")
        for ntyp in self._g.ntypes:
            print(f"{ntyp}({self._g.num_nodes(ntype=ntyp)}) ", end="")
        print()

        print(f"\tNumEdges: ", end="")
        for etyp in self._g.etypes:
            print(f"{etyp}({self._g.num_edges(etype=etyp)}) ", end="")
        print()

        print("\tNumFeats: ", end="")
        for ntyp in self._g.ntypes:
            print(f"{ntyp}:({self._g.ndata['x'][ntyp].shape[1]})", end="")
        print()

        print(f"\tNumClasses: {self.num_classes}")
        print(f"\tTargetType: {self.target_type}")

    def save(self):
        save_graphs(str(self.graph_path), self._g)

    def load(self):
        graphs, _ = load_graphs(str(self.graph_path))
        self._g = graphs[0]
        label = self._g.ndata["y"][self.target_type]
        self._num_classes = np.unique(label).size(0)

    def __getitem__(self, idx):
        assert idx == 0, f"dataset {self.name} just have one graph, but idx is {idx}"
        if self._transform is None:
            return self._g
        else:
            return self._transform(self._g)

    def __len__(self):
        return 1


class ACMDataset(HGMAEDataset):
    node_types = ["paper", "author", "subject"]

    num_nodes_dict = {
        "author": 7167,
        "paper": 4019,
        "subject": 60,
    }

    target_type = "paper"

    def __init__(self, raw_dir=None, force_reload=False, verbose=False, transform=None):
        super().__init__("acm", raw_dir, force_reload, verbose, transform)


class DBLPDataset(HGMAEDataset):
    node_types = ["paper", "author", "term", "conference"]

    num_nodes_dict = {
        "author": 4057,
        "paper": 14328,
        "term": 7723,
        "conference": 20,
    }

    target_type = "author"

    def __init__(self, raw_dir=None, force_reload=False, verbose=False, transform=None):
        super().__init__("dblp", raw_dir, force_reload, verbose, transform)


class FreebaseDataset(HGMAEDataset):
    node_types = ["movie", "director", "actor", "producer"]

    num_nodes_dict = {
        "movie": 3492,
        "director": 2502,
        "actor": 33401,
        "producer": 4459,
    }

    target_type = "movie"

    def __init__(self, raw_dir=None, force_reload=False, verbose=False, transform=None):
        super().__init__("freebase", raw_dir, force_reload, verbose, transform)


class IMDBDataset(HGMAEDataset):
    node_types = ["movie", "direct", "actor"]

    num_nodes_dict = {
        "movie": 4278,
        "direct": 2081,
        "actor": 5257,
    }

    target_type = "movie"

    def __init__(self, raw_dir=None, force_reload=False, verbose=False, transform=None):
        super().__init__("imdb", raw_dir, force_reload, verbose, transform)


class AminerDataset(HGMAEDataset):
    node_types = ["paper", "author", "reference"]

    num_nodes_dict = {
        "author": 13329,
        "paper": 6564,
        "reference": 35890,
    }

    target_type = "paper"

    def __init__(self, raw_dir=None, force_reload=False, verbose=False, transform=None):
        super().__init__("aminer", raw_dir, force_reload, verbose, transform)

