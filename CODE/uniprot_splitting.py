import os
import numpy as np
import shutil
import zipfile
import pickle


# Get all uniprot IDs

emb_name="onehot"
data_folder="../DATA/holo4k preprocessed/"
with zipfile.ZipFile(data_folder+f'/{emb_name}.zip') as thezip:
    all_files=[x.replace(emb_name+"_","") for x in thezip.namelist()]
    all_unps=np.unique([x.replace(emb_name,"")[8:][:-2] for x in thezip.namelist()])


# Split by UniProt ID
from sklearn.model_selection import train_test_split,KFold
model_unps,test_unps=train_test_split(all_unps,test_size=0.1)

n_folds=5
splitter=KFold(n_splits=n_folds,shuffle=True)
train_folds=[]
val_folds=[]
for train_idx,val_idx in splitter.split(model_unps):
    train_folds.append(model_unps[train_idx])
    val_folds.append(model_unps[val_idx])


# Group graph files by UniProt ID 
for k in range(n_folds):
    with open(f"../DATA/holo4k splits/holo4k_fold_{k}_train.txt","w") as f:
        for file in all_files:
            for unp_id in train_folds[k]:
                if unp_id in file:
                    f.write(file)
                    f.write("\n")
    
    with open(f"../DATA/holo4k splits/holo4k_fold_{k}_val.txt","w") as f:
        for file in all_files:
            for unp_id in val_folds[k]:
                if unp_id in file:
                    f.write(file)
                    f.write("\n")
    
with open(f"../DATA/holo4k splits/holo4k_test.txt","w") as f:
    for file in all_files:
        for unp_id in test_unps:
            if unp_id in file:
                f.write(file)
                f.write("\n")

