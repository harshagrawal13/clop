# CLOP
CLOP: Contrastive Learning for PPIs (the acronym doesn't properly fit).

## Setup
This repository is directly built upon ESM. Setup a new environment and install esm through [this documentation](https://github.com/facebookresearch/esm/tree/main/examples/inverse_folding#recommended-environment).

The following commands were used (for Linux Machine in an HPC cluster)

```
conda create -n esmfold python=3.9
conda activate esmfold

# Pytorch Installation with Torch Version = 1.12.1 & CUDA Toolkit=11.3
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

# Other Torch Sub-Dependencies
pip install -q torch-scatter -f https://data.pyg.org/whl/torch-{TORCH}+{CUDA}.html
pip install -q torch-sparse -f https://data.pyg.org/whl/torch-{TORCH}+{CUDA}.html
pip install -q torch-cluster -f https://data.pyg.org/whl/torch-{TORCH}+{CUDA}.html
pip install -q torch-spline-conv -f https://data.pyg.org/whl/torch-{TORCH}+{CUDA}.html
pip install -q torch-geometric

# ESM 
!pip install -q git+https://github.com/facebookresearch/esm.git

# Biotite
!pip install -q biotite

# Lightning
!pip install lightning

# GPU Nvidea Management
!pip install nvsmi
```
