"""
Download and Unzip CIF files from the PDB
CIF files were downloaded using the following RSYNC command:

rsync -rlpt -v -z --delete --port=33444 \
rsync.rcsb.org::ftp_data/structures/divided/mmCIF/ ./mmCIF

Other download alternatives are present here: https://www.wwpdb.org/ftp/pdb-ftp-sites
"""

# PUT THIS FILE IN THE BASE DIRECTORY OF THIS REPOSITORY.

import os
import gzip
import shutil

# Update the base cif download directory to your location of the downloaded files
cif_download_dir = "mmCIF/"

subdirs = os.listdir(cif_download_dir)
for subdir in subdirs:
    complete_dir = os.path.join(os.getcwd(), cif_download_dir, subdir)
    expand_dir = os.listdir(complete_dir)

    # Unzip all files
    for file in expand_dir:
        if file[-3:] == ".gz":
            try:
                complete_file_path = os.path.join(complete_dir, file)
                with gzip.open(complete_file_path, "rb") as f_in:
                    with open(complete_file_path[:-3], "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)

                # Delete the .gz files after unzipping
                os.remove(complete_file_path)

            except Exception as e:
                # Some files are corrupted and cannot be unzipped
                print(e)
                print(file)
