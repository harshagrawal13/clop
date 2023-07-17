from os import path
import numpy as np
import json
import pickle
from typing import Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch.core import LightningDataModule

from esm.inverse_folding import util
from esm import Alphabet

DEFAULT_NUM_WORKERS = 1
MAX_SEQ_LEN = 200
DEFAULT_SPLIT_RATIO = 0.8
DEFAULT_TRAIN_SHUFFLE = True
DEFAULT_VAL_SHUFFLE = False
DEFAULT_PIN_MEMORY = False
DEFUALT_SPLIT = "train"
DEFAULT_DATA_DIR = path.join(path.dirname(path.abspath(__file__)), "data/")


class ESMDataset(Dataset):
    def __init__(
        self,
        data_dir=DEFAULT_DATA_DIR,
        max_seq_len=MAX_SEQ_LEN,
        split=DEFUALT_SPLIT,
        **kwargs,
    ):
        """ESM Dataset: torch.utils.data.Dataset

        Args:
            data_dir (str): Data Directory. Defaults to DEFAULT_DATA_DIR.
            dataset_type (str): Dataset type. Defaults to "train".
            max_seq_len (int): Max Sequence Length. Defaults to 200.
        """
        with open(path.join(data_dir, "all_structures.pkl"), "rb") as f:
            self.structure_data = pickle.load(f)

        with open(path.join(data_dir, "all_seqs.json"), "r") as f:
            self.sequence_data = json.load(f)

        with open(path.join(data_dir, "all_prot_names.json"), "r") as f:
            self.prot_names = json.load(f)
        self.filter_data(max_seq_len)

        # Split data
        split_ratio = kwargs.get("split_ratio", DEFAULT_SPLIT_RATIO)
        # return train/valid/test data
        self.split_data(split, split_ratio)

    def split_data(self, split, split_ratio):
        assert split in [
            "train",
            "val",
        ], f"Split: {split} must be train/val"

        # Data Split Index = Split Ratio * Total Dataset Length
        data_split_idx = split_ratio * self.__len__()

        if split == "train":
            self.structure_data = self.structure_data[: int(data_split_idx)]
            self.sequence_data = self.sequence_data[: int(data_split_idx)]
            self.prot_names = self.prot_names[: int(data_split_idx)]
        else:
            self.structure_data = self.structure_data[int(data_split_idx) :]
            self.sequence_data = self.sequence_data[int(data_split_idx) :]
            self.prot_names = self.prot_names[int(data_split_idx) :]

    def filter_data(self, max_seq_len: int):
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

    def __getitem__(self, idx) -> Tuple[np.ndarray, None, str]:
        """Returns the idx-th protein in the dataset

        Args:
            idx (Protein index): Protein Index

        Returns:
            Tuple[np.ndarray, None, str]: Protein Structure Data, None, Protein Sequence Data
        """
        # chunk where the idx-th protein is located
        protein_structure = self.structure_data[idx]
        protein_sequence = self.sequence_data[idx]
        # protein_name = self.currently_opened_chunk["prot_names"][protein_idx]
        return protein_structure, None, protein_sequence


class ESMDataLoader(DataLoader):
    def __init__(
        self,
        esm2_alphabet: Alphabet,
        esm_if_alphabet: Alphabet,
        dataset: ESMDataset,
        batch_size: int,
        shuffle: int,
        num_workers=DEFAULT_NUM_WORKERS,
        **kwargs
    ):
        """ESM DataLoader

        Args:
            esm2_alphabet (Alphabet): ESM-2 Alphabet
            esm_if_alphabet (Alphabet): ESM-IF Alphabet
            data_dir (str): Data Directory. Defaults to DEFAULT_DATA_DIR.
            batch_size (int): Batch Size. Defaults to 64.
            shuffle (bool, optional): Shuffle Dataset. Defaults to True.
            num_workers (int, optional): Num Workers to process dataset. Defaults to 1.
        """
        self.esm2_alphabet = esm2_alphabet
        self.esm_if_alphabet = esm_if_alphabet

        self.esm_if_batch_converter = util.CoordBatchConverter(
            self.esm_if_alphabet
        )
        self.esm_2_batch_converter = self.esm2_alphabet.get_batch_converter()

        # self.collate_fn = util.CoordBatchConverter(alphabet)
        self.dataset = dataset
        super().__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            pin_memory=kwargs.get("pin_memory", DEFAULT_PIN_MEMORY),
        )

    def collate_fn(
        self, batch
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
        _labels, _strs, tokens = self.esm_2_batch_converter(inp_seqs)

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
        batch_size: int,
        **kwargs,
    ):
        super().__init__()
        self.esm2_alphabet = esm2_alphabet
        self.esm_if_alphabet = esm_if_alphabet

        self.esm_if_batch_converter = util.CoordBatchConverter(
            self.esm_if_alphabet
        )
        self.esm_2_batch_converter = self.esm2_alphabet.get_batch_converter()

        self.batch_size = batch_size
        self.data_dir = kwargs.get("data_dir", DEFAULT_DATA_DIR)
        self.max_seq_len = kwargs.get("max_seq_len")
        self.split_ratio = kwargs.get("split_ratio", DEFAULT_SPLIT_RATIO)
        self.train_shuffle = kwargs.get("train_shuffle", DEFAULT_TRAIN_SHUFFLE)
        self.val_shuffle = kwargs.get("val_shuffle", DEFAULT_VAL_SHUFFLE)
        self.num_workers = kwargs.get("num_workers", DEFAULT_NUM_WORKERS)
        self.pin_memory = kwargs.get("pin_memory", DEFAULT_PIN_MEMORY)

    def prepare_data(self):
        """Load Train and Val Dataset"""
        self.train_dataset = ESMDataset(
            self.data_dir, split="train", split_ratio=self.split_ratio, max_seq_len=self.max_seq_len
        )
        self.val_dataset = ESMDataset(
            self.data_dir, split="val", split_ratio=self.split_ratio, max_seq_len=self.max_seq_len
        )

    def setup(self, stage):
        self.train_loader = ESMDataLoader(
            esm2_alphabet=self.esm2_alphabet,
            esm_if_alphabet=self.esm_if_alphabet,
            dataset=self.train_dataset,
            shuffle=self.train_shuffle,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

        self.val_loader = ESMDataLoader(
            esm2_alphabet=self.esm2_alphabet,
            esm_if_alphabet=self.esm_if_alphabet,
            dataset=self.val_dataset,
            shuffle=self.val_shuffle,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
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
