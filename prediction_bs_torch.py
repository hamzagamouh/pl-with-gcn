import argparse
from re import M

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default=3000, type=int, help="Epochs.")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--layers", default=[1024, 2048, 4096, 1024], nargs='+', type=int, help="Number of Convolutional layers")
parser.add_argument("--emb_name", default="onehot", type=str, help="Embedding name : bert, albert, onehot, xlnet")
parser.add_argument("--evaluate_each", default=10, type=str, help="Evaluate each number of epochs")

print("HELLO !")
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

if torch.cuda.is_available():
    print("CUDA is available !!")
else:
    print("CUDA is NOT available !!")

args = parser.parse_args([] if "__file__" not in globals() else None)
# 
import warnings
warnings.filterwarnings("ignore")


# 
from sklearn.metrics import recall_score,accuracy_score,precision_score
from sklearn.metrics import matthews_corrcoef as MCC


def get_metric(pred,labels,name):
    
    # Accuracy
    if name=="accuracy":
        return accuracy_score(np.array(labels.cpu()), np.array(pred.cpu()))
    
    # Recall
    if name=="recall":
        return recall_score(np.array(labels.cpu()), np.array(pred.cpu()),average='macro')

    # Precision
    if name=="precision":
        return precision_score(np.array(labels.cpu()), np.array(pred.cpu()),average='macro')
    # Matthews Correlation Coefficient (MCC)
    if name=="mcc":
        return MCC(np.array(labels.cpu()), np.array(pred.cpu()))




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
            h = F.relu(h)
            h = F.dropout(h)

        h=self.convs[-1](g, h)
        h = F.softmax(h,1)
        return h

##__________________ Load data and make batches _____________________________________

# Data folder
data_folder="//./app/data"
EMB_NAME=args.emb_name
FEAT_SIZES={"onehot":21,
            "bert":1024,
            "albert":4096,
            "xlnet":1024}



def get_graphs(mode):
    with zipfile.ZipFile(data_folder+f'/{EMB_NAME}_{mode}_batched_{args.batch_size}.zip') as thezip:
        return thezip.namelist()

GRAPHS={mode:get_graphs(mode) for mode in ["train","val","test"]}


# def get_batch(a,mode):
#     b=a+args.batch_size
#     # print(f"{mode} batches : {a} - {b}")
#     with zipfile.ZipFile(data_folder+f'/{EMB_NAME}_{mode}.zip') as thezip:
#         for k,graph in enumerate(GRAPHS[mode][a:b]):
#             g=pickle.load(thezip.open(graph,"r"))
#             g.ndata["feat"]=g.ndata["feat"]
#             g.ndata["label"]=g.ndata["label"]
#             if k==0:
#                 batch_graph=g
#             else:
#                 batch_graph=dgl.batch([batch_graph,g])
            

#     batch_graph=batch_graph.to('cuda:0')
#     features = batch_graph.ndata['feat'].to('cuda:0')
#     labels = batch_graph.ndata['label'].to('cuda:0')
#     n0=0
#     n1=0
#     if mode=="train":
#         n0=int((labels==0).sum())
#         n1=int((labels==1).sum())

#     return batch_graph,features,labels,n0,n1


def get_batch(k,mode):
    # print(f"{mode} batches : {a} - {b}")
    with zipfile.ZipFile(data_folder+f'/{EMB_NAME}_{mode}_batched_{args.batch_size}.zip') as thezip:
        graph=GRAPHS[mode][k]
        # print(graph)
        batch_graph=pickle.load(thezip.open(graph,"r"))

    batch_graph=batch_graph.to('cuda:0')
    features = batch_graph.ndata['feat'].to('cuda:0')
    labels = batch_graph.ndata['label'].to('cuda:0')
    n0=0
    n1=0
    if mode=="train":
        n0=int((labels==0).sum())
        n1=int((labels==1).sum())

    return batch_graph,features,labels,n0,n1

# Get results

RESULTS={"Epoch":[],"Train MCC":[],"Val MCC":[]}



# Train the model    

def train(model,metric_name):
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    for e in range(args.epochs+1):
        evaluate=(e%args.evaluate_each==0)

        if evaluate:
            train_pred=torch.Tensor([]).to("cuda:0")
            train_labels=torch.Tensor([]).to("cuda:0")
        a_train=0
        # while a_train<len(GRAPHS["train"]):
        for a_train in range(len(GRAPHS["train"])):
            # load batch
            g_train,features,labels,n0,n1=get_batch(a_train,"train")
            # Forward propagation
            logits = model(g_train, features)
            if evaluate:
                # Get predictions and labels for evaluation
                train_pred = torch.concat([train_pred,logits.argmax(1)]) 
                train_labels = torch.concat([train_labels,labels]) 
            # Compute loss
            # Note that you should only compute the losses of the nodes in the training set.
            weight=(torch.Tensor([n1,n0])).to('cuda:0')
            loss=nn.CrossEntropyLoss(weight)(logits.float(), labels.reshape(-1,).long())
            # Backward propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # a_train+=args.batch_size
        
        random.shuffle(GRAPHS["train"])

        if evaluate:
            # print("Loading validation batches...")
            val_pred=torch.tensor([]).to("cuda:0")
            val_labels=torch.tensor([]).to("cuda:0")
            # a_val=0
            # while a_val<len(GRAPHS["val"]):
            for a_val in range(len(GRAPHS["val"])):
                # load batch
                g_val,features,labels,_,_=get_batch(a_val,"val")
                # Forward propagation
                logits = model(g_val, features)
                # Get predictions and labels for evaluation
                val_pred = torch.concat([val_pred,logits.argmax(1)]) 
                val_labels = torch.concat([val_labels,labels])
                # Increment counter
                # a_val+=args.batch_size


            # Get evaluation metric       
            train_metric=get_metric(train_pred,train_labels,metric_name)
            val_metric=get_metric(val_pred,val_labels,metric_name)
            RESULTS["Epoch"].append(e+1)
            RESULTS["Train MCC"].append(train_metric)
            RESULTS["Val MCC"].append(val_metric)

            print('In epoch {}, loss: {:.3f}, train {} : {:.3f} , val {} : {:.3f}'.format(
                e, loss,metric_name,train_metric,metric_name ,val_metric))
            
            
        


# Configure training
print("BEGIN TRAINING ")
n_feats=FEAT_SIZES[EMB_NAME]

layers=[n_feats]+args.layers+[2]

print("Embedding : ",EMB_NAME)
print("Layers : ",layers)
print("Batch size : ",args.batch_size)
model = GCN(layers).cuda()

# Train the model
train(model,metric_name="mcc")

# Test on test set
print("Evaluating on test set ...")
test_pred=torch.tensor([]).to("cuda:0")
test_labels=torch.tensor([]).to("cuda:0")
# a_test=0
# while a_test<len(GRAPHS["test"]):
for a_test in range(len(GRAPHS["test"])):
    # load batch
    g_test,features,labels,_,_=get_batch(a_test,"test")
    # Forward propagation
    logits = model(g_test, features)
    # Get predictions and labels for evaluation
    test_pred = torch.concat([test_pred,logits.argmax(1)]) 
    test_labels = torch.concat([test_labels,labels])
    # Increment counter
    # a_test+=args.batch_size

print("Test MCC ",get_metric(test_pred,test_labels,"mcc"))


# Save the results
# layers_string=""
# for l in layers:
#     layers_string+=str(l)+"_"

# pd.DataFrame(RESULTS).to_csv(f"//./app/results/{EMB_NAME}_{layers_string}.csv")



