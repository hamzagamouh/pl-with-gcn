print("TESTING THE FINAL MODEL")
import os
import time
import numpy as np
import pandas as pd
import pickle
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import shutil
import zipfile
import random
import scipy

if torch.cuda.is_available():
    print("CUDA is available !!")
else:
    print("CUDA is NOT available !!")

# 
import warnings
warnings.filterwarnings("ignore")


# 
from sklearn.metrics import recall_score,accuracy_score,precision_score,confusion_matrix,f1_score
from sklearn.metrics import matthews_corrcoef as MCC


def get_metric(pred,labels,name):
    # Accuracy
    if name=="f1-score":
        return f1_score(np.array(labels.cpu()), np.array(pred.cpu()))
    
    # Recall
    if name=="recall":
        return recall_score(np.array(labels.cpu()), np.array(pred.cpu()))

    # Precision
    if name=="precision":
        return precision_score(np.array(labels.cpu()), np.array(pred.cpu()))
    # Matthews Correlation Coefficient (MCC)
    if name=="mcc":
        return MCC(np.array(labels.cpu()), np.array(pred.cpu()))
    
    if name=="confusion":
        return confusion_matrix(np.array(labels.cpu()), np.array(pred.cpu()))

##__________________ Load data and make batches _____________________________________

# Data folder
data_folder="../DATA/holo4k_batched"
EMB_NAME="xlnet"
FEAT_SIZES={"onehot":21,
            "bert":1024,
            "xlnet":1024}

no_graph=""
TEST_MODE="test"
MODES=[TEST_MODE]
batch_size=32

batch_norm=True
dropout=True
dropout_rate=0.3
n_feats=FEAT_SIZES[EMB_NAME]

# _____________________________ Model Architecture ______________________________
from dgl.nn import GraphConv as ConvLayer
class GCN(nn.Module):
    
    def __init__(self, layers):
        super(GCN, self).__init__()
        self.convs=nn.ModuleList([ConvLayer(layers[k], layers[k+1]) for k in range(len(layers)-1)])  

    def forward(self, g, in_feat):
        for k,layer in enumerate(self.convs[:-1]):
            if k==0:
                h=in_feat
            h = layer(g, h)
            if batch_norm:
                h=torch.nn.BatchNorm1d(h.shape[1]).to("cuda:0")(h)
            h = F.relu(h)
            if dropout:
                h=F.dropout(h,dropout_rate)

        h=self.convs[-1](g, h)
        h = F.softmax(h,1)
        return h


def get_graphs(mode):
    with zipfile.ZipFile(data_folder+f'/{EMB_NAME}_{mode}{no_graph}_batched_{batch_size}.zip') as thezip:
        return thezip.namelist()

GRAPHS={mode:get_graphs(mode) for mode in MODES}




def get_batch(k,mode):
    with zipfile.ZipFile(data_folder+f'/{EMB_NAME}_{mode}{no_graph}_batched_{batch_size}.zip') as thezip:
        graph=GRAPHS[mode][k]
        batch_graph=pickle.load(thezip.open(graph,"r"))
    

    batch_graph=batch_graph.to('cuda:0')
    features = batch_graph.ndata['feat'].to('cuda:0')
    labels = batch_graph.ndata['label'].to('cuda:0')
    w0=0
    w1=0
    return batch_graph,features,labels,w0,w1




layers=[n_feats]+[512]*6+[2]
model = GCN(layers).cuda()
model.load_state_dict(torch.load("../RESULTS/xlnet_model_final.pt"))
model.eval()
# # Test on test set
print("Evaluating on test set ...")
test_pred=torch.tensor([]).to("cuda:0")
test_labels=torch.tensor([]).to("cuda:0")
for a_test in range(len(GRAPHS[TEST_MODE])):
    # load batch
    g_test,features,labels,_,_=get_batch(a_test,TEST_MODE)
    # Forward propagation
    logits = model(g_test, features)
    # Get predictions and labels for evaluation
    test_pred = torch.concat([test_pred,logits.argmax(1)]) 
    test_labels = torch.concat([test_labels,labels])

with open("/RESULTS/confusion_matrix.txt","w") as file:

    tn, fp, fn, tp=get_metric(test_pred,test_labels,"confusion").ravel()
    file.write(f"TN : {tn}   FP : {fp}   FN : {fn}   TP : {tp}")
    file.write("\n")
    file.write(f"Precision : {get_metric(test_pred,test_labels,'precision')}")
    file.write("\n")
    file.write(f"Sensitivity/recall : {get_metric(test_pred,test_labels,'recall')}")
    file.write("\n")
    file.write(f"F1-measure : {get_metric(test_pred,test_labels,'f1-score')}")
    file.write("\n")
    file.write(f"MCC : {get_metric(test_pred,test_labels,'mcc')}")
    file.write("\n")






