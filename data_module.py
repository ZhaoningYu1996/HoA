import os
import lightning as L
from torch_geometric.loader import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset

class OGBDataModule(L.LightningDataModule):
    def __init__(self, data_name: str = "ogbg-molhiv", data_dir: str = "path/to/dir", batch_size: int = 1):
        super().__init__()
        self.data_name = data_name
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.dataset = PygGraphPropPredDataset(name=self.data_name, root=self.data_dir)
        split_idx = self.dataset.get_idx_split()
        self.feat_dim = self.dataset[0].x.size(1)
        self.out_dim = self.dataset.num_tasks
        self.train_set = self.dataset[split_idx["train"]]
        self.val_set = self.dataset[split_idx["valid"]]
        self.test_set = self.dataset[split_idx["test"]]
        print(f"Number of training graphs: {len(self.train_set)}")
        print(f"Number of validation graphs: {len(self.val_set)}")
        print(f"Number of test graphs: {len(self.test_set)}")

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=os.cpu_count(), shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=os.cpu_count())

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=os.cpu_count())