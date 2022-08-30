# Protein-Ligand binding sites prediction using Graph Convolutional Networks (GCN)


This is the code repository for the master's thesis : Protein-Ligand binding sites prediction using Graph Convolutional Networks (GCN)


### Original dataset 
The original dataset is the HOLO4K dataset which contains pdb files. You can find it in `/DATA/holo4k`

### Preprocessing datasets
For each of the embedding types : `onehot` , `ProtTransBERT` and `ProtTransXLNET`, we preprocessed the pdb files to get chains with the same UniProt ID. The preprocessed data is a set of `dgl.graph` objects for a set of chains which share the same PDB ID and UniProt ID. 

The preprocessing script can be found in `/CODE/protein_bs_preprocessing.py`

You can find preprocessed datasets in `/DATA/holo4k_preprocessed` in `.zip` format.

### Train/Val/Test splitting
For splitting the dataset, we performed a random train/val/test split by UniProt IDs, and we grouped the inputs by their UniProt ID. We used the script `/CODE/uniprot_splitting.py` 

### Making batches for training and testing
To make the training faster, we made batches of size 32 of the preprocessed graphs using the script `/CODE/batch_generator.py`

### Training and Evaluation
For training and evaluation, we used the following script `/CODE/training_bs.py` 
