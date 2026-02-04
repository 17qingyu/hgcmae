import os.path as osp
from typing import Type, Union

from dgl.dataloading import GraphDataLoader
from dgl.transforms import BaseTransform
from hydra.utils import get_class
from lightning import LightningDataModule
import scipy.sparse as sp
import torch

from .dataset import HGMAEDataset
from .utils import sparse_mx_to_torch_tensor


class PretrainDataModule(LightningDataModule):
    def __init__(self,
                 data_class: Union[str, Type[HGMAEDataset]],
                 root: str,
                 transform: BaseTransform = None,
                 batch_size: int = 1,
                 num=0):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.transform = transform
        if isinstance(data_class, str):
            data_class = get_class(data_class)
        self.data_class = data_class
        self.num = num

    def prepare_data(self):
        self.data_class(raw_dir=self.root)

    def setup(self, stage=None):
        self.dataset = self.data_class(
            raw_dir=self.root, transform=self.transform)

    def _collate_fn_train(self, batch):
        batch_graph = batch[0] if len(batch) == 1 else batch
        pos_path = self.dataset.save_dir
        if not self.num:
            pos = torch.load(osp.join(pos_path, "pos.th"))
        else:
            pos = torch.load(osp.join(pos_path, f"pos_{self.num}.th"))
        return batch_graph, self.dataset.target_type, pos

    def _collate_fn_test(self, batch):
        batch_graph = batch[0] if len(batch) == 1 else batch
        return batch_graph, self.dataset.target_type

    def train_dataloader(self):
        return GraphDataLoader(self.dataset, batch_size=self.batch_size, collate_fn=self._collate_fn_train)

    def val_dataloader(self):
        return GraphDataLoader(self.dataset, batch_size=self.batch_size, collate_fn=self._collate_fn_test)

    def test_dataloader(self):
        return GraphDataLoader(self.dataset, batch_size=self.batch_size, collate_fn=self._collate_fn_test)


class TrainDataModule(LightningDataModule):
    def __init__(self,
                 data_class: Union[str, Type[HGMAEDataset]],
                 root: str,
                 transform: BaseTransform = None,
                 batch_size: int = 1,
                 num_train_node: int = 60):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.transform = transform
        if isinstance(data_class, str):
            data_class = get_class(data_class)
        self.data_class = data_class
        self.num_train_node = num_train_node

    def prepare_data(self):
        self.data_class(raw_dir=self.root, transform=self.transform)

    def setup(self, stage=None):
        self.dataset = self.data_class(
            raw_dir=self.root, transform=self.transform)
        traget_type = self.dataset.target_type
        self.dataset[0].nodes[traget_type].data["train_mask"] = (
            self.dataset[0].nodes[traget_type].data[f"train_{self.num_train_node}_mask"])
        self.dataset[0].nodes[traget_type].data["val_mask"] = (
            self.dataset[0].nodes[traget_type].data[f"val_{self.num_train_node}_mask"])
        self.dataset[0].nodes[traget_type].data["test_mask"] = (
            self.dataset[0].nodes[traget_type].data[f"test_{self.num_train_node}_mask"])

    def _collate_fn(self, batch):
        batch_graph = batch[0] if len(batch) == 1 else batch
        return batch_graph

    def train_dataloader(self):
        return GraphDataLoader(self.dataset, batch_size=self.batch_size, collate_fn=self._collate_fn)

    def val_dataloader(self):
        return GraphDataLoader(self.dataset, batch_size=self.batch_size, collate_fn=self._collate_fn)

    def test_dataloader(self):
        return GraphDataLoader(self.dataset, batch_size=self.batch_size, collate_fn=self._collate_fn)
