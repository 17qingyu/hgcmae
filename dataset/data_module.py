from typing import Type, Union

from dgl.dataloading import GraphDataLoader
from dgl.transforms import BaseTransform
from hydra.utils import get_class
from lightning import LightningDataModule

from .dataset import HGMAEDataset


class PretrainDataModule(LightningDataModule):
    def __init__(
        self,
        data_class: Union[str, Type[HGMAEDataset]],
        root: str,
        k: int,
        pos_num: int,
        transform: BaseTransform = None,
        batch_size: int = 1,
        force_reload: bool = False,
    ):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.transform = transform
        if isinstance(data_class, str):
            data_class = get_class(data_class)
        self.data_class = data_class
        self._k = k
        self._pos_num = pos_num
        self.force_reload = force_reload


    def prepare_data(self):
        devices = self.trainer.device_ids
        
        if devices and len(devices) > 0:
            self._device = f"cuda:{devices[0]}"
        else:
            self._device = "cpu"

        self.data_class(
            k=self._k, pos_num=self._pos_num, device=self._device, raw_dir=self.root
        )

    def setup(self, stage=None):
        self.dataset = self.data_class(
            k=self._k,
            pos_num=self._pos_num,
            device=self._device,
            raw_dir=self.root,
            transform=self.transform,
            force_reload=self.force_reload,
        )

    def _collate_fn_train(self, batch):
        batch_graph = batch[0] if len(batch) == 1 else batch
        return batch_graph, self.dataset.target_type

    def _collate_fn_test(self, batch):
        batch_graph = batch[0] if len(batch) == 1 else batch
        return batch_graph, self.dataset.target_type

    def train_dataloader(self):
        return GraphDataLoader(
            self.dataset, batch_size=self.batch_size, collate_fn=self._collate_fn_train
        )

    def val_dataloader(self):
        return GraphDataLoader(
            self.dataset, batch_size=self.batch_size, collate_fn=self._collate_fn_test
        )

    def test_dataloader(self):
        return GraphDataLoader(
            self.dataset, batch_size=self.batch_size, collate_fn=self._collate_fn_test
        )


class TrainDataModule(LightningDataModule):
    def __init__(
        self,
        data_class: Union[str, Type[HGMAEDataset]],
        root: str,
        transform: BaseTransform = None,
        batch_size: int = 1,
        num_train_node: int = 60,
    ):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.transform = transform
        if isinstance(data_class, str):
            data_class = get_class(data_class)
        self.data_class = data_class
        self.num_train_node = num_train_node


    def prepare_data(self):
        self.device = f"cuda:{self.trainer.device_ids[0]}"
        self.data_class(
            raw_dir=self.root,
            k=2,
            pos_num=7,
            device=self.device,
            transform=self.transform,
        )

    def setup(self, stage=None):
        self.dataset = self.data_class(
            raw_dir=self.root,
            k=2,
            pos_num=7,
            device=self.device,
            transform=self.transform,
        )
        traget_type = self.dataset.target_type
        self.dataset[0].nodes[traget_type].data["train_mask"] = (
            self.dataset[0].nodes[traget_type].data[f"train_{self.num_train_node}_mask"]
        )
        self.dataset[0].nodes[traget_type].data["val_mask"] = (
            self.dataset[0].nodes[traget_type].data[f"val_{self.num_train_node}_mask"]
        )
        self.dataset[0].nodes[traget_type].data["test_mask"] = (
            self.dataset[0].nodes[traget_type].data[f"test_{self.num_train_node}_mask"]
        )

    def _collate_fn(self, batch):
        batch_graph = batch[0] if len(batch) == 1 else batch
        return batch_graph

    def train_dataloader(self):
        return GraphDataLoader(
            self.dataset, batch_size=self.batch_size, collate_fn=self._collate_fn
        )

    def val_dataloader(self):
        return GraphDataLoader(
            self.dataset, batch_size=self.batch_size, collate_fn=self._collate_fn
        )

    def test_dataloader(self):
        return GraphDataLoader(
            self.dataset, batch_size=self.batch_size, collate_fn=self._collate_fn
        )
