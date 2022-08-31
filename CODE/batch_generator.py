import os
import numpy as np
import pandas as pd
import pickle
import dgl
import torch
import zipfile


p2rank_test=list(pd.read_csv("../DATA/p2rank_test.csv")["pdb_id"])
data_folder="../DATA/holo4k preprocessed"
batch_size=32

def get_batch(a,mode,emb_name,graphs):
    b=a+batch_size
    print(f"{mode} batch : {a} - {b}")
    with zipfile.ZipFile(data_folder+f'/{emb_name}.zip') as thezip:
        for k,graph in enumerate(graphs[a:b]):
            g=pickle.load(thezip.open(graph,"r"))
            if k==0:
                batch_graph=g
            else:
                batch_graph=dgl.batch([batch_graph,g])
    
    return batch_graph

# Batch data for train/val/test splits
def batch_data(mode,emb_name):
    with open(f"../DATA/holo4k splits/holo4k_{mode}.txt") as f:
        graphs=[f"{emb_name}_{x.strip()}.p" for x in f.readlines()]
    with zipfile.ZipFile(f'../DATA/holo4k_batched/{emb_name}_{mode}_batched_{batch_size}.zip',"w") as thezip:
        a=0
        while a<len(graphs):
            # load batch
            graph=get_batch(a,mode,emb_name,graphs)
            filename=f"{emb_name}_{mode}_{a}_{a+batch_size}.p"
            pickle.dump(graph,open(filename,"wb"))
            thezip.write(filename,filename,compress_type=zipfile.ZIP_BZIP2)
            os.remove(filename)
            a+=batch_size

# Batch data to compare with p2rank
def batch_data_p2rank(mode,emb_name):
    with open(f"../DATA/holo4k splits/holo4k_{mode}.txt") as f:
        graphs=[f"{emb_name}_{x.strip()}.p" for x in f.readlines() if x[:4] in p2rank_test]
    with zipfile.ZipFile(f'../DATA/holo4k_batched/{emb_name}_{mode}_p2rank_batched_{batch_size}.zip',"w") as thezip:
        a=0
        while a<len(graphs):
            # load batch
            graph=get_batch(a,mode,emb_name,graphs)
            filename=f"{emb_name}_{mode}_{a}_{a+batch_size}.p"
            pickle.dump(graph,open(filename,"wb"))
            thezip.write(filename,filename,compress_type=zipfile.ZIP_BZIP2)
            os.remove(filename)
            a+=batch_size



# Create batches for training and evaluation
for EMB_NAME in ["onehot","bert","xlnet"]:
    print(EMB_NAME)
    batch_data("test",EMB_NAME)
    for k in range(5):
        print(f"FOLD {k} TRAIN _______________________________________")
        batch_data(f"fold_{k}_train",EMB_NAME)
        batch_data(f"fold_{k}_val",EMB_NAME)


# Create batches for comparing the best model with p2rank
for EMB_NAME in ["xlnet"]:
    print(EMB_NAME)
    batch_data_p2rank("model_dev",EMB_NAME)
    batch_data_p2rank("test",EMB_NAME)
    