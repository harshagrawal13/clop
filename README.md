# JESPR
JESPR: Joint Embedding Pre-training for Protein Structure-Sequence Representations (the acronym doesn't quite fit). 

__The code is still under construction and relevant results still haven't been obtained.__

The motivation for JESPR was derived from FAIR's (Meta) class of ESM models [[1]](https://github.com/facebookresearch/esm) and OpenAI's CLIP [[2]](https://github.com/openai/CLIP). Protein Sequence &rarr; Structure and Protein Sequence &rarr; Structure prediction are generally worked upon as two separate tasks. As presented by Meta's ESM, ESM-Fold (model responsible for Sequence &rarr; Structure) and  ESM-Inverse-Fold are two separate models that are built using the protein LLM trained on a masked modelling objective millions of Uniprot sequences.

We want to question whether Protein Folding/Protein Inverse Folding can be solved by one unified pre-training. 

Moreover, we believe that auto-regressive next residue prediction is a poor pre-training task for proteins as it forces the model to predict nitty-gritty details in the sequence space (which is pseudo-infinite as compared to the English language vocabulary which is ~50K). This doesn't directly encourage the model to learn higher-level semantic features and thus necessitate larger models. Thus, we are using encoder-only models to represent the sequence and 3-d structure in a high-dimensional joint-embedding space with the objective to minimize the distance between their embeddings. We want to experiment if even such a simple pre-training objective can prove to be better at learning protein semantics than the current set of foundational models.

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
