import os
import numpy as np
import json

import torch
from torch.utils.data import Dataset, DataLoader

import esm
from esm.inverse_folding import util, gvp_transformer
from esm import Alphabet, ESM2


class ESMDataset(Dataset):
    def __init__(self, chunks_dir="data/chunks", chunk_size=100):
        self.chunk_size = chunk_size
        self.chunks_dir = chunks_dir
        self.num_chunks = len(os.listdir(chunks_dir))
        self.currently_opened_chunk = {
            "chunk_idx": 0,
            "structure": None,
            "sequence": None,
            "prot_names": None,
        }

    def __len__(self):
        return self.num_chunks * self.chunk_size

    def open_chunk(self, chunk_idx):
        print("Opening Chunk: ", chunk_idx)
        self.currently_opened_chunk["chunk_idx"] = chunk_idx

        chunk_dir = os.path.join(os.getcwd(), self.chunks_dir, str(chunk_idx))
        with open(os.path.join(chunk_dir, "structure.json"), "r") as f:
            self.currently_opened_chunk["structure"] = json.load(f)

        with open(os.path.join(chunk_dir, "seq.json"), "r") as f:
            self.currently_opened_chunk["sequence"] = json.load(f)

        with open(os.path.join(chunk_dir, "prot_names.json"), "r") as f:
            self.currently_opened_chunk["prot_names"] = json.load(f)

    def __getitem__(self, idx):
        """Returns the idx-th protein in the dataset

        Args:
            idx (Protein index): Protein Index

        Returns:
            tuple: A tuple containing the
            protein structure, confidence (None), and the protein sequence
        """
        # chunk where the idx-th protein is located
        chunk_location = idx // self.chunk_size + 1
        if chunk_location != self.currently_opened_chunk["chunk_idx"]:
            self.open_chunk(chunk_location)

        protein_idx = idx % self.chunk_size
        protein_structure = np.array(
            self.currently_opened_chunk["structure"][protein_idx],
            dtype=np.float32,
        )
        # print(protein_structure.shape)
        protein_sequence = self.currently_opened_chunk["sequence"][protein_idx]
        protein_name = self.currently_opened_chunk["prot_names"][protein_idx]
        return protein_structure, None, protein_sequence


class ESMDataLoader(DataLoader):
    def __init__(
        self,
        alphabet: Alphabet,
        chunk_dir="data/chunks",
        chunk_size=100,
        batch_size=64,
        shuffle=True,
        num_workers=1,
    ):
        self.collate_fn = util.CoordBatchConverter(alphabet)
        self.dataset = ESMDataset(chunks_dir=chunk_dir, chunk_size=chunk_size)
        super().__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
        )
