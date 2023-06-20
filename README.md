# JESPR
JESPR: Joint Embedding Pre-training for Protein Structure-Sequence Representations (the acronym doesn't quite fit). 

The motivation for JESPR was derived from FAIR's (Meta) class of ESM models [[1]](https://github.com/facebookresearch/esm) and OpenAI's CLIP [[2]](https://github.com/openai/CLIP). Protein Sequence &rarr; Structure and Protein Sequence &rarr; Structure prediction are generally worked upon as two seperate tasks. As presented by Meta's ESM, ESM-Fold (model responsible for Sequence &rarr; Structure) and  ESM-Inverse-Fold are two seperate models that are built using the protein LLM trained on a masked modelling objective millions of Uniprot sequences.

We want to question whether Protein Folding/Protein Inverse Folding are two seperate tasks or can they be solved togther by one unified pre-training. 

We try two approaches to tackle our problem:
- Contrastive Pre-Training to obtain joint embeddings for protein structure and sequences.
- Non-Contrastive, Latent Variable Pre-Training to obtain joint embeddings for protein structure and sequences.


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