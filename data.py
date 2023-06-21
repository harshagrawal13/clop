import os
import numpy as np
import json
import pickle

from torch.utils.data import Dataset, DataLoader

from esm.inverse_folding import util
from esm import Alphabet


class ESMDataset(Dataset):
    def __init__(self, data_dir="data/chunks", chunk_size=100):
        data_dir_complete = os.path.join(os.getcwd(), data_dir)
        with open(
            os.path.join(data_dir_complete, "all_structures.pkl"), "rb"
        ) as f:
            self.structure_data = pickle.load(f)

        with open(os.path.join(data_dir_complete, "all_seqs.json"), "r") as f:
            self.sequence_data = json.load(f)

        with open(
            os.path.join(data_dir_complete, "all_prot_names.json"), "r"
        ) as f:
            self.prot_names = json.load(f)

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
        alphabet: Alphabet,
        data_dir="data/",
        batch_size=64,
        shuffle=True,
        num_workers=1,
    ):
        self.alphabet = alphabet
        self.collate_fn = util.CoordBatchConverter(alphabet)
        self.dataset = ESMDataset(data_dir=data_dir)
        super().__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            pin_memory=False,
        )
