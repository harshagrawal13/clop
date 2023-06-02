from lightning.pytorch.core import LightningDataModule
from data import ESMDataLoader


class MyDataModule(LightningDataModule):
    def __init__(self, esm_data_loader: ESMDataLoader):
        super().__init__()
        self.esm_data_loader = esm_data_loader

    def prepare_data(self):
        pass
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        pass

    def train_dataloader(self):
        return self.esm_data_loader

    def val_dataloader(self):
        return self.esm_data_loader

    def test_dataloader(self):
        return self.esm_data_loader

    def teardown(self):
        # clean up after fit or test
        # called on every process in DDP
        self.esm_data_loader = None
