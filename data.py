from os import path
import numpy as np
import json
import pickle
from typing import Tuple
from argparse import Namespace

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from lightning.pytorch.core import LightningDataModule

from esm.inverse_folding import util
from esm import Alphabet


class ESMDataset(Dataset):
    def __init__(self, args: Namespace) -> None:
        """ESM Dataset: torch.utils.data.Dataset

        Args:
            args (Namespace): Args for ESMDataset. Must Contain:
                - data_dir (str): Data Directory
                - max_seq_len (int): Max Sequence Length
        """
        with open(path.join(args.data_dir, "all_structures.pkl"), "rb") as f:
            self.structure_data = pickle.load(f)

        with open(path.join(args.data_dir, "all_seqs.json"), "r") as f:
            self.sequence_data = json.load(f)

        with open(path.join(args.data_dir, "all_prot_names.json"), "r") as f:
            self.prot_names = json.load(f)

        # filter data by sequence length
        if args.max_seq_len is not None:
            self.filter_data(args.max_seq_len)

    def filter_data(self, max_seq_len: int) -> None:
        """Filter the dataset by sequence length

        Args:
            max_seq_len (int): Maximum sequence length
        """
        structure_data = []
        sequence_data = []
        prot_names = []
        for i, seq in enumerate(self.sequence_data):
            if len(seq) <= max_seq_len:
                structure_data.append(self.structure_data[i])
                sequence_data.append(seq)
                prot_names.append(self.prot_names[i])

        # update the dataset
        self.structure_data = structure_data
        self.sequence_data = sequence_data
        self.prot_names = prot_names

    def __len__(self) -> int:
        """Returns the length of the dataset

        Returns:
            _type_: _description_
        """
        return len(self.prot_names)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, None, str]:
        """Returns the idx-th protein in the dataset

        Args:
            idx (int): Protein Index

        Returns:
            Tuple[np.ndarray, None, str]: Protein Structure Data, None, Protein Sequence Data
        """
        # chunk where the idx-th protein is located
        return self.structure_data[idx], None, self.sequence_data[idx]


class ESMDataLoader(DataLoader):
    def __init__(
        self,
        esm2_alphabet: Alphabet,
        esm_if_alphabet: Alphabet,
        dataset: ESMDataset,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        # pin_memory: bool,
        **kwargs
    ):
        """ESM DataLoader

        Args:
            esm2_alphabet (Alphabet): ESM-2 Alphabet
            esm_if_alphabet (Alphabet): ESM-IF Alphabet
            dataset (ESMDataset): ESMDataset.
            batch_size (int): Batch Size
            shuffle (bool): Shuffle
            num_workers (int): Number of Workers
            pin_memory (bool): Pin Memory
        """
        self.esm2_alphabet = esm2_alphabet
        self.esm_if_alphabet = esm_if_alphabet

        self.esm_if_batch_converter = util.CoordBatchConverter(
            self.esm_if_alphabet
        )
        self.esm2_batch_converter = self.esm2_alphabet.get_batch_converter()

        # self.collate_fn = util.CoordBatchConverter(alphabet)

        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            # pin_memory=False,
        )

    def collate_fn(
        self, batch: list
    ) -> Tuple[torch.tensor, torch.tensor, list, torch.tensor, torch.tensor]:
        """
        Collate Function to process each batch
        through ESM-IF CoordBatch Converter and ESM2 Batch Converter

        Args:
            batch (list): List of individual items from dataset.__getitem__()

        Returns:
            tuple: coords, confidence, strs, tokens, padding_mask
        """
        # Prepare input seqs for esm2 batch converter as mentioned in
        # the example here: https://github.com/facebookresearch/esm/blob/2b369911bb5b4b0dda914521b9475cad1656b2ac/README.md?plain=1#L176
        inp_seqs = [("", item[2]) for item in batch]

        # Process ESM-2 ->
        _labels, _strs, tokens = self.esm2_batch_converter(inp_seqs)

        # Process ESM-IF ->
        (
            coords,
            confidence,
            strs,
            _,
            padding_mask,
        ) = self.esm_if_batch_converter(batch)

        return coords, confidence, strs, tokens, padding_mask


class ESMDataLightning(LightningDataModule):
    def __init__(
        self,
        esm2_alphabet: Alphabet,
        esm_if_alphabet: Alphabet,
        args: Namespace,
    ) -> None:
        """Initialize Lightning DataModule class for JESPR

        Args:
            esm2_alphabet (Alphabet): ESM-2 Alphabet
            esm_if_alphabet (Alphabet): ESM-IF Alphabet
            args (Namespace): Args. Must contain:
                - data_dir (str): Data Directory
                - split_ratio (int): Dataset split ratio. Eg: 0.8 (80% train, 20% val)
                - max_seq_len (int): Max Sequence Length
                - batch_size (int): Batch Size
                - train_shuffle (bool): Train Shuffle
                - train_num_workers (int): Train Loader - Number of Workers
                - train_pin_memory (bool): Train Loader - Pin Memory
                - val_shuffle (bool): Val Shuffle
                - val_num_workers (int): Val Loader - Number of Workers
                - val_pin_memory (bool): Val Loader - Pin Memory

        """
        super().__init__()
        self.esm2_alphabet = esm2_alphabet
        self.esm_if_alphabet = esm_if_alphabet

        self.esm_if_batch_converter = util.CoordBatchConverter(
            self.esm_if_alphabet
        )
        self.esm2_batch_converter = self.esm2_alphabet.get_batch_converter()
        self.args = args

    def prepare_data(self):
        pass

    def setup(self, stage):
        """Load Train and Val Dataset"""
        dataset = ESMDataset(args=self.args)
        data_len = dataset.__len__()
        split_idx = int(data_len * self.args.split_ratio)
        self.train_dataset, self.val_dataset = random_split(
            dataset,
            [split_idx, data_len - split_idx],
        )
        self.train_loader = ESMDataLoader(
            esm2_alphabet=self.esm2_alphabet,
            esm_if_alphabet=self.esm_if_alphabet,
            dataset=self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=self.args.train_shuffle,
            num_workers=self.args.train_num_workers,
            # pin_memory=self.args.train_pin_memory,
        )

        self.val_loader = ESMDataLoader(
            esm2_alphabet=self.esm2_alphabet,
            esm_if_alphabet=self.esm_if_alphabet,
            dataset=self.val_dataset,
            batch_size=self.args.batch_size,
            shuffle=self.args.val_shuffle,
            num_workers=self.args.val_num_workers,
            # pin_memory=self.args.val_pin_memory,
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.val_loader

    def teardown(self, stage):
        # clean up after fit or test
        # called on every process in DDP
        self.train_loader = None
        self.val_loader = None
