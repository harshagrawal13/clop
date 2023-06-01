"""
Process mmCIF files and save into JSON chunks for input the model.
"""

from esm.inverse_folding.util import load_structure, extract_coords_from_structure
import time, os
import json
import shutil

all_structures = []
all_seqs = []
all_prot_names = []

import string
ALL_CHAINS = list(string.ascii_uppercase)

def check_and_append(complete_file_path, chain_id):
    structure = load_structure(complete_file_path, chain_id)
    coords, native_seq = extract_coords_from_structure(structure)

    if native_seq not in all_seqs:
        all_structures.append(coords.tolist())
        all_seqs.append(native_seq)
        file_name = f"{complete_file_path[-8:-4]}-{chain_id}"
        all_prot_names.append(file_name)

def save_chunk(chunk_num):
    try:
        print("Saving chunk...", chunk_num)
        chunk_dir = os.path.join(os.getcwd(), "data/chunks/", str(chunk_num))
        if os.path.exists(chunk_dir):
            shutil.rmtree(chunk_dir)

        os.makedirs(chunk_dir)
        with open(os.path.join(chunk_dir, "structure.json"), "w") as f:
            json.dump(all_structures, f, indent=4)
            print("Saved", len(all_structures), "structures")
        with open(os.path.join(chunk_dir, "seq.json"), "w") as f:
            json.dump(all_seqs, f, indent=4)
            
        with open(os.path.join(chunk_dir, "prot_names.json"), "w") as f:
            json.dump(all_prot_names, f, indent=4)

        all_structures.clear()
        all_seqs.clear()
        all_prot_names.clear()

    except Exception as e:
        print(e)

if __name__ == "__main__":
    print("Starting program...")

    chunk_counter = 1
    
    # Change accordingly
    chunk_size = 100

    subdirs = os.listdir("data/mmcif/")
    start_time = time.perf_counter()
    for subdir in subdirs:
        complete_dir = os.path.join(os.getcwd(), "data/mmcif/", subdir)
        
        expand_dir = os.listdir(complete_dir)

        all_file_paths = [os.path.join(complete_dir, file_) for file_ in expand_dir if file_[-4:] == ".cif"]
        
        for file in all_file_paths:
            for chain_id in ALL_CHAINS:
                try:
                    check_and_append(file, chain_id)
                    if len(all_prot_names) == chunk_size:
                        save_chunk(chunk_counter)
                        chunk_counter += 1

                except Exception as e:
                    # print(e)
                    break

    finish_time = time.perf_counter()
    print(f"Program finished in {finish_time-start_time} seconds")