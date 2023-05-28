# JESPR
JESPR: Joint Embedding Pre-training for Protein Structure-Sequence Representations (the acronym doesn't quite fit). 

The motivation for JESPR was derived from FAIR's (Meta) class of ESM models [[1]](https://github.com/facebookresearch/esm) and OpenAI's CLIP [[2]](https://github.com/openai/CLIP). Protein Sequence &rarr; Structure and Protein Sequence &rarr; Structure prediction are generally worked upon as two seperate tasks. As presented by Meta's ESM, ESM-Fold (model responsible for Sequence &rarr; Structure) and  ESM-Inverse-Fold are two seperate models that are built using the protein LLM trained on a masked modelling objective millions of Uniprot sequences.

We want to question whether Protein Folding/Protein Inverse Folding are two seperate tasks or can they be solved togther by one unified pre-training. 

We try two approaches to tackle our problem:
- Contrastive Pre-Training to obtain joint embeddings for protein structure and sequences.
- Non-Contrastive, Latent Variable Pre-Training to obtain joint embeddings for protein structure and sequences.
