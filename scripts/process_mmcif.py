import json
import numpy as np
import biotite
from biotite.structure.io import pdbx
from esm.inverse_folding.util import extract_coords_from_structure
from biotite.structure import filter_backbone
from biotite.structure import get_chains

import time, os
from os import path
import shutil
import string
import pickle

all_seqs = []
all_prot_names = []
all_structures = []

data_dir = os.path.join(path.abspath(path.join(__file__, "../..")), "data/")


def save():
    with open(os.path.join(data_dir, "all_seqs.json"), "w") as f:
        json.dump(all_seqs, f, indent=4)

    with open(os.path.join(data_dir, "all_prot_names.json"), "w") as f:
        json.dump(all_prot_names, f, indent=4)

    with open(os.path.join(data_dir, "all_structures.pkl"), "wb") as f:
        pickle.dump(all_structures, f)


def load_structure(fpath):
    """
    Args:
        fpath: filepath to either pdb or cif file
        chain: the chain id or list of chain ids to load
    Returns:
        biotite.structure.AtomArray
    """
    if fpath.endswith("cif"):
        with open(fpath) as fin:
            pdbxf = pdbx.PDBxFile.read(fin)
        structure = pdbx.get_structure(pdbxf, model=1)

    bbmask = filter_backbone(structure)
    structure = structure[bbmask]
    if len(structure) == 0:
        return (None, None)

    all_chains = get_chains(structure)
    if len(all_chains) == 0:
        raise ValueError("No chains found in the input file.")

    return structure, all_chains


def main(save_every=2000):
    for subdir in sorted(os.listdir(os.path.join(data_dir, "mmCIF/"))):
        complete_dir = os.path.join(data_dir, "mmCIF/", subdir)
        expand_dir = sorted(os.listdir(complete_dir))

        all_file_paths = [
            os.path.join(complete_dir, f)
            for f in expand_dir
            if f.endswith(".cif")
        ]

        for file in all_file_paths:
            structure, chain_ids = load_structure(file)
            if chain_ids is not None:
                for chain_id in chain_ids:
                    try:
                        chain_filter = [
                            atom_.chain_id == chain_id for atom_ in structure
                        ]
                        coords, native_seq = extract_coords_from_structure(
                            structure[chain_filter]
                        )

                        if native_seq not in all_seqs:
                            all_seqs.append(native_seq)
                            all_structures.append(coords)
                            all_prot_names.append(f"{file[-8:-4]}-{chain_id}")
                    except Exception as e:
                        print(e)
                        break

            if len(all_prot_names) % save_every == 0:
                print(f"Saving; current length = {len(all_prot_names)}")
                save()

    # Save everything at the end
    print("Saving last chunk...")
    save()


if __name__ == "__main__":
    start_time = time.time()
    main(save_every=10000)
    end_time = time.time()
    print(f"Time taken = {end_time - start_time}")
