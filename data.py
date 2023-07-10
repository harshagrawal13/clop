from os import path
import numpy as np
import json
import pickle
from typing import Tuple
import torch
from torch.utils.data import Dataset, DataLoader

from esm.inverse_folding import util
from esm import Alphabet


class ESMDataset(Dataset):
    def __init__(self, data_dir="data", max_seq_len=200):
        data_dir_complete = path.join(
            path.abspath(path.join(__file__, "..")), data_dir
        )
        with open(
            path.join(data_dir_complete, "all_structures.pkl"), "rb"
        ) as f:
            self.structure_data = pickle.load(f)

        with open(path.join(data_dir_complete, "all_seqs.json"), "r") as f:
            self.sequence_data = json.load(f)

        with open(
            path.join(data_dir_complete, "all_prot_names.json"), "r"
        ) as f:
            self.prot_names = json.load(f)
        self.filter_data(max_seq_len)

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

    def __len__(self):
        return len(self.prot_names)

    def __getitem__(self, idx):
        """Returns the idx-th protein in the dataset

        Args:
            idx (Protein index): Protein Index

        Returns:
            tuple: A tuple containing the
            protein structure, confidence (None), and the protein sequence
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
        data_dir="data/",
        batch_size=64,
        shuffle=True,
        num_workers=1,
    ):
        """ESM DataLoader

        Args:
            esm2_alphabet (Alphabet): ESM-2 Alphabet
            esm_if_alphabet (Alphabet): ESM-IF Alphabet
            data_dir (str): Data Directory. Defaults to "data/".
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
        self.dataset = ESMDataset(data_dir=data_dir)
        super().__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            pin_memory=False,
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
