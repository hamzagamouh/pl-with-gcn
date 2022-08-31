from fileinput import filename
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Bio.PDB import *
import pickle
import dgl
import torch
import scipy
import shutil
import zipfile

import warnings
warnings.filterwarnings("ignore")


# Amino acid codons
AMINO_ACIDS={'ALA':'A', 'ARG':'R', 'ASN':'N', 'ASP':'D', 'CYS':'C', 'GLN':'Q', 'GLU':'E',
             'GLY':'G', 'HIS':'H', 'ILE':'I', 'LEU':'L',
                 'LYS':'K', 'MET':'M', 'PHE':'F', 'PRO':'P', 'SER':'S'
             , 'THR':'T', 'TRP':'W', 'TYR':'Y', 'VAL':'V',
             'PYL':'X','SEC':'X','UNK':'X'} #21 


ACs=list(AMINO_ACIDS.keys())
AC_letters=list(AMINO_ACIDS.values())


# PDB parser
parser = PDBParser()

# SeqIO parser (getting sequences)
from Bio import SeqIO

##### Useful functions

# Check if a residue is an amino acid
def is_AC(res):
    return res.get_full_id()[3][0]==" "

# Check if a residue is a ligand
def is_ligand(res):
    return res.get_full_id()[3][0] not in ["W"," "]

# Get residues from chains
def get_residues(chains):
    residues=[]
    for chain in chains:
        for res in chain:
            if is_AC(res):
                residues.append(res)
    return residues

# Get ligands from chains
def get_ligands(chains):
    ligands=[]
    for chain in chains:
        for res in chain:
            if is_ligand(res):
                ligands.append(res)
    return ligands

# Get AA sequence
def get_sequence(chain):
    s=""
    for res in get_residues([chain]):
        s+=AMINO_ACIDS[res.get_resname()]
    return s

# Check if 2 residues are connected  (2 residues are connected if their alpha carbon atoms are close enough)
def are_connected(res1,res2,th):
    # Check all atoms in the first residue and get their coordinates
    for atom1 in res1.get_unpacked_list():
        coord1=atom1.get_coord()
        # if some atom is a central carbon atom take its coordinates
        if atom1.get_name()=="CA":
            break
            
    for atom2 in res2.get_unpacked_list():
        coord2=atom2.get_coord()
        # if some atom is a central carbon atom take its coordinates
        if atom2.get_name()=="CA":
            break
    
    # Check distance
    distance=np.linalg.norm(coord1-coord2)
    if distance<th:
        return 1
    
    return 0

# Check if a residue is a binding site
def is_binding_site(res,ligand,th):
    for atom in res.get_unpacked_list():
        atom_coord=atom.get_coord()
        for lig_atom in ligand:
            lig_coord=lig_atom.get_coord()
            distance=np.linalg.norm(lig_coord-atom_coord)
            if distance<th:
                return 1
    return 0


# This function parses a protein and returns
# 1. an array of the protein's adjacency matrix (residue nodes)
# 2. an embedding for the sequence of residues in the protein (features)
# 3. an array of labels representing ligandability of a residue to a ligand

def get_graph(chains):
    residues=get_residues(chains) 
    # Adjacency matrix at the residue level
    n_res=len(residues)
    A = np.zeros((n_res,n_res))  
    for i in range(n_res):
        for j in range(n_res):
            A[i,j]=are_connected(residues[i],residues[j],th=6)   # Threshold = 6 Angstroms
    # Get all atoms of all ligands
    ligands=get_ligands(chains)
    ligs_atoms=[]
    for lig in ligands:
        ligs_atoms+=lig.get_unpacked_list()
    
    
    # Labels represent the binding site status of the residue
    labels = np.zeros((n_res,1))  
    for i in range(n_res):
        labels[i]=is_binding_site(residues[i],ligs_atoms,th=4)    # Threshold = 4 Angstroms
    
    print("Number of ligands : ",len(ligands))
    print((labels==1).sum()," binding sites of ",n_res)
    return scipy.sparse.csr_matrix(A),labels

def get_embeddings(chains,embedder):
    # Get the sequence embeddings
    for k,chain in enumerate(chains):
        sequence=get_sequence(chain)
        if k==0:
            emb=embedder.embed(sequence)
        else:
            emb=np.concatenate(emb,embedder.embed(sequence),axis=0)
    return emb

def create_dgl_graph(chains):
    A,labels=get_graph(chains)
    g=dgl.from_scipy(A)
    g.ndata["label"]=torch.tensor(labels).long()
    return g


def create_dgl_data(g,feats):
    g.ndata["feat"]=torch.tensor(feats).long()
    return g


def process_file(data_folder,file):
    prot_file=os.path.join(data_folder,file)
    proteins={}

    # Get the SEQIO sequences with uniprot and pdb IDs
    with open(prot_file) as handle:
        seqio_chains={}
        for rec in SeqIO.parse(handle, "pdb-seqres"):
            pdb_id=rec.id[:-2]
            if rec.dbxrefs!=[]:
                uniprot_id = rec.dbxrefs[0]
                if "UNP" not in uniprot_id:
                    uniprot_id="UNKOWN"
                else:
                    uniprot_id=uniprot_id[4:]
            else:
                uniprot_id="UNKOWN"
            
            if uniprot_id=="UNKOWN":
                continue

            chain_id=rec.id[-1]

            if pdb_id+" - UNP "+uniprot_id in seqio_chains.keys():
                seqio_chains[pdb_id+" - UNP "+uniprot_id]+=[chain_id]
            else:
                seqio_chains[pdb_id+" - UNP "+uniprot_id]=[chain_id]

    
    # Get the PDB structure and get the chains from it
    structure = parser.get_structure([file][:-4],prot_file) 

    for chain in structure.get_chains():
        for key,ids in seqio_chains.items():
            if chain.id in ids:
                if key in proteins.keys():
                    proteins[key]+=[chain]
                else:
                    proteins[key]=[chain]
    
    return proteins


# Process files with uniprot IDs from PDB database

uniprot_from_pdb=pd.read_csv("../DATA/uniprot from pdb.csv")

def get_uniprot_id(pdb_id,chain_id):
    df=uniprot_from_pdb[(uniprot_from_pdb["PDB ID"]==pdb_id) & (uniprot_from_pdb['Chain ID'].str.contains(chain_id))]
    try :
        return list(df["UniProt ID"])[0]
    except:
        return "UNKNOWN"


def process_file_uniprot(data_folder,file):
    prot_file=os.path.join(data_folder,file)
    proteins={}

    # Get the SEQIO sequences with uniprot and pdb IDs
    with open(prot_file) as handle:
        seqio_chains={}
        for rec in SeqIO.parse(handle, "pdb-seqres"):
            pdb_id=rec.id[:-2]
            chain_id=rec.id[-1]
            uniprot_id=get_uniprot_id(pdb_id,chain_id)
            if uniprot_id=="UNKNOWN":
                continue

            if pdb_id+" - UNP "+uniprot_id in seqio_chains.keys():
                seqio_chains[pdb_id+" - UNP "+uniprot_id]+=[chain_id]
            else:
                seqio_chains[pdb_id+" - UNP "+uniprot_id]=[chain_id]

    
    # Get the PDB structure and get the chains from it
    structure = parser.get_structure([file][:-4],prot_file) 

    for chain in structure.get_chains():
        for key,ids in seqio_chains.items():
            if chain.id in ids:
                if key in proteins.keys():
                    proteins[key]+=[chain]
                else:
                    proteins[key]=[chain]
    
    return proteins 

##### _____________________PREPROCESSING FILES_______________________________ 
# Preprocess files and get graphs and chains
data_folder="../DATA/holo4k/"

with open("../DATA//holo4k_preprocessed/all_chains_with_seq.csv","w") as f:
    for k,file in enumerate(os.listdir(data_folder)):
        if k%100==0:
            print(k)
        prots=process_file(data_folder,file)
        for key,chains in prots.items():
            for chain in chains:
                f.write(key+","+chain.id+","+get_sequence(chain))
                f.write("\n")
        g=create_dgl_graph(chains)

        pickle.dump(g,open(f"../DATA/holo4k_preprocessed/graphs/{key}.p","wb"))
        with open(f"../DATA/holo4k_preprocessed/sequences/{key}.txt","w") as f1:
            for chain in chains:
                f1.write(get_sequence(chain))
                f1.write("\n")


##### ____________________ Fill in missing uniprot IDs______________________
# This is a script to handle missing uniprot IDs, please comment the above script and execute this one for missing uniprot IDs

# with open("../DATA/uniprots.txt") as f1:
#     with open("../DATA/holo4k_preprocessed/all_chains_missing.csv","w") as f:
#         for k,line in enumerate(f1.readlines()):
#             file=line.strip().lower()+".pdb"
#             if k%100==0:
#                 print(k)
#             prots=process_file_uniprot(data_folder,file)
#             for key,chains in prots.items():
#                 for chain in chains:
#                     f.write(key+","+chain.id+","+get_sequence(chain))
#                     f.write("\n")
#             # g=create_dgl_graph(chains)
#             # pickle.dump(g,open(f"../DATA/holo4k_preprocessed/graphs/{key}.p","wb"))
# #             # with open(f"../DATA/holo4k_preprocessed/sequences/{key}.txt","w") as file:
# #             #     for chain in chains:
# #             #         file.write(get_sequence(chain))
# #             # #         file.write("\n")



##### _____________________Get embeddings and convert to dgl graph_______________________________ 
# Bio embedding configuration
EMB_NAME="onehot"

# # Embedders
# Onehot embedding
if EMB_NAME=="onehot":
    from bio_embeddings.embed.one_hot_encoding_embedder import OneHotEncodingEmbedder
    EMBEDDER=OneHotEncodingEmbedder()
# XLNET 
if EMB_NAME=="xlnet":
    from bio_embeddings.embed.prottrans_xlnet_uniref100_embedder import ProtTransXLNetUniRef100Embedder
    EMBEDDER=ProtTransXLNetUniRef100Embedder()
# BERT 
if EMB_NAME=="bert":
    from bio_embeddings.embed.prottrans_bert_bfd_embedder import ProtTransBertBFDEmbedder
    EMBEDDER=ProtTransBertBFDEmbedder()


seq_folder="../DATA/holo4k_preprocessed/sequences"
graphs_folder="../DATA/holo4k_preprocessed/graphs"

# seq_folder="/app/data/new_seqs"
# graphs_folder="/app/data/new_graphs"


with zipfile.ZipFile(f'../DATA/holo4k_preprocessed/{EMB_NAME}.zip',"w") as thezip:
    for file in os.listdir(seq_folder):
        file=file[:-4]
        with open(os.path.join(seq_folder,file+".txt"),"r") as f:
            # print(f"{file} ---> Getting embeddings...")
            for k,sequence in enumerate(f.readlines()):
                sequence=sequence.strip()
                new_feat=EMBEDDER.embed(sequence)
                if k==0:
                    feats=new_feat
                else:
                    feats=np.concatenate((feats,new_feat),axis=0)
        
        # print(f"{file} ---> Constructing graph...")
        g=pickle.load(open(os.path.join(graphs_folder,file+".p"),'rb'))
        g=create_dgl_data(g,feats)

        assert g.ndata["label"].shape[0]==g.ndata["feat"].shape[0]

        # print(f"{file} ---> Adding to archive...")
        filename=f"../DATA/holo4k_preprocessed/{EMB_NAME}_{file}.p"
        pickle.dump(g,open(filename,"wb"))
        thezip.write(filename,f"{EMB_NAME}_{file}.p",compress_type=zipfile.ZIP_BZIP2)
        os.remove(filename)
