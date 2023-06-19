from esm.inverse_folding.util import (
    load_structure,
    extract_coords_from_structure,
)

import time, os
import json
import shutil
import string

all_seqs = []
all_prot_names = []
all_structures = []

ALL_CHAINS = list(string.ascii_uppercase)

current_dir = os.getcwd()
subdirs = os.listdir("/rds/general/user/at3722/home/atom51/data/mmCIF/")

data_dir = os.path.join(current_dir, "atom51/jespr/data/")

if not os.path.exists(data_dir):
    os.mkdir(data_dir)


def main():
    for subdir in subdirs:
        complete_dir = os.path.join(current_dir, "atom51/data/mmCIF/", subdir)
        expand_dir = os.listdir(complete_dir)

        all_file_paths = [
            os.path.join(complete_dir, file_)
            for file_ in expand_dir
            if file_[-4:] == ".cif"
        ]
        print("complete dir:", complete_dir)

        for file in all_file_paths:
            complete_file_path = os.path.join(complete_dir, file)
            for chain_id in ALL_CHAINS:
                try:
                    struct = load_structure(complete_file_path, chain_id)
                    coords, native_seq = extract_coords_from_structure(struct)
                    if native_seq not in all_seqs:
                        all_seqs.append(native_seq)
                        all_structures.append(coords.tolist())
                        all_prot_names.append(
                            f"{complete_file_path[-8:-4]}-{chain_id}"
                        )
                except Exception as e:
                    break

            if len(all_prot_names) % 1000 == 0:
                print("Saving")
                with open(os.path.join(data_dir, "all_seqs.json"), "w") as f:
                    json.dump(all_seqs, f, indent=4)

                with open(
                    os.path.join(data_dir, "all_prot_names.json"), "w"
                ) as f:
                    json.dump(all_prot_names, f, indent=4)

                with open(
                    os.path.join(data_dir, "all_structures.json"), "w"
                ) as f:
                    json.dump(all_structures, f, indent=4)


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Time taken = {end_time - start_time}")
