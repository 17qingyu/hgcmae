from dgl.transforms import BaseTransform
import numpy as np
import torch

from .utils import sample_per_class, get_train_val_test_split


class DatasetSplit(BaseTransform):
    def __init__(self, 
        num_train_per_class=None, 
        num_val_per_class=None, 
        num_test_per_class=None, 
        train_size=None,
        val_size=None,
        test_size=None,
        seed=0
    ):
        super().__init__()
        self.num_train_per_class = num_train_per_class
        self.num_val_per_class = num_val_per_class
        self.num_test_per_class = num_test_per_class
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.seed = seed

    def __call__(self, g):
        target_type = None
        for ntyp in g.ntypes:
            if 'y' in g.nodes[ntyp].data:
                target_type = ntyp
                break

        labels = g.nodes[target_type].data['y']
        num_nodes = labels.shape[0]

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        # train_indices = sample_per_class(
        #     labels, self.num_train_per_class, seed=self.seed)
        # val_indices = sample_per_class(
        #     labels, self.num_val_per_class, forbidden_indices=train_indices, seed=self.seed)
        # test_indices = sample_per_class(labels, self.num_test_per_class, forbidden_indices=torch.hstack([
        #                                 train_indices, val_indices]), seed=self.seed)
        random_state = np.random.RandomState(seed=self.seed)
        train_indices, val_indices, test_indices = get_train_val_test_split(
            random_state, 
            labels, 
            self.num_train_per_class, 
            self.num_val_per_class, 
            self.num_test_per_class, 
            self.train_size,
            self.val_size, 
            self.test_size
        )

        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True

        g.nodes[target_type].data[f'train_{self.num_train_per_class}_mask'] = train_mask
        g.nodes[target_type].data[f'val_{self.num_train_per_class}_mask'] = val_mask
        g.nodes[target_type].data[f'test_{self.num_train_per_class}_mask'] = test_mask

        return g
