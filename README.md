# Protein-Ligand binding sites prediction using Graph Convolutional Networks (GCN)


This is the code repository for the master's thesis : **_Protein-Ligand binding sites prediction using Graph Convolutional Networks (GCN)_**


### Environment
To reproduce the results, create a conda environment `conda create --name pl_bs_pred python=3.7`
And install the pip requirements using `pip install -r requirements.txt` . Please note that `dgl-cu111` is suitable for CUDA version 11. If you don't have GPU, please consider installing `dgl` instead (See `requirements.txt` file).

### Original dataset 
The original dataset is the HOLO4K dataset which contains pdb files. You can download it using the following link https://drive.google.com/file/d/17qqtsnmpw4NqAs-FLvNG92MQtbBu8g07/view?usp=sharing 
When you want to work with scripts, you can download it and extract it in `/DATA/`.

### Preprocessing datasets
For each of the embedding types : `onehot` , `ProtTransBERT` and `ProtTransXLNET`, we preprocessed the pdb files to get chains with the same UniProt ID. The preprocessed data is **a set of `dgl.graph` objects** for a set of Amino Acid chains which share the same PDB ID and UniProt ID. 

The preprocessing script can be found in `/CODE/protein_bs_preprocessing.py`

You can download the preprocessed dataset is `.zip` format :
    - Onehot embedding : https://drive.google.com/file/d/18NdpUh-RHJgF6YdxftlOOEqjpokBvpeS/view?usp=sharing
    - ProtTransBERT embedding : https://drive.google.com/file/d/1o-x_8Z2GRhqqABlalrvW2CvpmiwRvKMC/view?usp=sharing
    - ProtTransXLNET embedding : https://drive.google.com/file/d/14tGO9T3fmfwx1JN9cyccd7QLRH16C9_y/view?usp=sharing

If you want to skip the preprocessing part, you can download the `.zip` files and move them to the folder `/DATA/holo4k_preprocessed`.

### Train/Val/Test splitting
For splitting the dataset, we performed a random train/val/test split by UniProt IDs, then we grouped the inputs by their UniProt ID. We used the script `/CODE/uniprot_splitting.py` . The generated splits (PDB,UniProt pairs) are to be found in `/DATA/holo4k_splits` 

### Making batches for training and testing
To make the training faster, we made batches of the preprocessed graphs using the script `/CODE/batch_generator.py`. You can execute this script once you have the preprocessed dataset in `.zip` format.

### Training and Evaluation
For training and evaluation, we used the following script `/CODE/training_bs.py` 

### Results of training
The learning curves of different models and configurations necessary to reproduce the plots in the thesis, as well as the best obtained model can be found in `/RESULTS/`.
To test the final model you can execute `/RESULTS/testing_final_model.py`