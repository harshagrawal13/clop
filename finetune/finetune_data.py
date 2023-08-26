from os import path
import pickle
from typing import Tuple
from argparse import Namespace


import torch
from esm.data import Alphabet
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch.core import LightningDataModule


class StabilityDataset(Dataset):
    def __init__(self, split: str, data_dir: str) -> None:
        """Stability Dataset

        Args:
            split (str): Split.
                One of train, val, test
            data_dir (str): Data directory

        Raises:
            ValueError: If Split is invalid
        """
        assert split in [
            "train",
            "val",
            "test",
        ], f"Invalid Split: {split}"

        with open(path.join(data_dir, f"stability/{split}.pkl"), "rb") as f:
            self.data = pickle.load(f)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple:
        """Get Item from Index

        Args:
            index (int): Index

        Returns:
            Tuple: (AA Sequence, Class Label)
        """
        entry = self.data[index]
        return (entry["primary"], entry["stability_score"][0])


class StabilityLightning(LightningDataModule):
    def __init__(self, esm2_alphabet: Alphabet, args: Namespace) -> None:
        """Definite Stability Lightning module

        Args:
            esm2_alphabet (Alphabet): ESM2 Alphabet for batch_converter
            args (Namespace): Args
        """
        super().__init__()
        self.args = args
        self.esm2_batch_converter = esm2_alphabet.get_batch_converter()

    def prepare_data(self):
        pass

    def setup(self, stage):
        if stage == "fit":
            self.train_dataset = StabilityDataset(
                split="train",
                data_dir=self.args.data_dir,
            )
            self.val_dataset = StabilityDataset(
                split="val",
                data_dir=self.args.data_dir,
            )
        elif stage == "test":
            self.test_dataset = StabilityDataset(
                split="test",
                data_dir=self.args.data_dir,
            )

    def collate_fn(self, batch: list) -> Tuple[str, int]:
        """
        Collate Function to process each batch
        through ESM2 Batch Converter

        Args:
            batch (list): List of individual items from dataset.__getitem__()

        Returns:
            tuple: tokens, labels
        """
        # Prepare input seqs for esm2 batch converter as mentioned in
        # the example here: https://github.com/facebookresearch/esm/blob/2b369911bb5b4b0dda914521b9475cad1656b2ac/README.md?plain=1#L176
        inp_seqs = [("", item[0]) for item in batch]
        stability_score = torch.tensor([item[1] for item in batch], dtype=torch.float32)

        # Process ESM-2 ->
        _labels, _strs, tokens = self.esm2_batch_converter(inp_seqs)

        return tokens, stability_score

    def train_dataloader(self) -> DataLoader:
        """Return Train Data Loader

        Returns:
            DataLoader: Train Dataloader
        """
        assert self.train_dataset is not None, "Setup not called with fit stage"
        dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=self.args.train_shuffle,
            num_workers=self.args.train_num_workers,
            collate_fn=self.collate_fn,
        )
        return dataloader

    def val_dataloader(self) -> DataLoader:
        """Return Val Data Loader

        Returns:
            DataLoader: Val Dataloader
        """
        assert self.val_dataset is not None, "Setup not called with fit stage"
        dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.args.batch_size,
            shuffle=self.args.train_shuffle,
            num_workers=self.args.train_num_workers,
            collate_fn=self.collate_fn,
        )
        return dataloader

    def test_dataloader(self, holdout: str) -> DataLoader:
        """Return Test Data Loader

        Args:
            holdout (str): One of family, superfamily, fold

        Returns:
            DataLoader: Test Dataloader
        """
        assert self.test_dataset is not None, "Setup not called with test stage"

        dataloader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.args.batch_size,
            shuffle=self.args.train_shuffle,
            num_workers=self.args.train_num_workers,
            collate_fn=self.collate_fn,
        )
        return dataloader

    def teardown(self, stage):
        # clean up after fit or test
        # called on every process in DDP
        if stage == "fit":
            self.train_dataloader = None
            self.val_dataloader = None
        elif stage == "test":
            self.test_dataloader = None
