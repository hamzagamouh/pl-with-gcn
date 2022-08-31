import argparse
from re import M

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default=1000, type=int, help="Epochs.")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--layers", default=[512]*6, nargs='+', type=int, help="Number of Convolutional layers")
parser.add_argument("--batch_norm", default=True, type=bool, help="Whether or not to use batch normalization")
parser.add_argument("--dropout", default=0.3, type=float, help="Dropout rate")
parser.add_argument("--residual", default=False, type=bool, help="Whether or not to use residual connections")
parser.add_argument("--no_graph", default=True, type=bool, help="Whether to remove structure information")
parser.add_argument("--no_emb", default=True, type=bool, help="Whether to remove feature information")
parser.add_argument("--emb_name", default="onehot", type=str, help="Embedding name : onehot, bert, xlnet")
parser.add_argument("--fold", default="fold_0_", type=str, help="Cross-Validation fold number (ex : fold_0_")
parser.add_argument("--evaluate_each", default=50, type=str, help="Evaluate each number of epochs")
parser.add_argument("--test_mode", default=True, type=bool, help="Whether to evaluate the model on test set")




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
import scipy

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
        h1=torch.tensor([[0]])
        for k,layer in enumerate(self.convs[:-1]):
            if k==0:
                h=in_feat
            h = layer(g, h)
            if args.batch_norm:
                h=torch.nn.BatchNorm1d(h.shape[1]).to("cuda:0")(h)
            h = F.relu(h)
            if args.dropout:
                h=F.dropout(h,args.dropout)
            if args.residual:
                if h1.shape[1]==h.shape[1]:
                    h=h+h1
                h1=h

        h=self.convs[-1](g, h)
        h = F.softmax(h,1)
        return h

##__________________ Load data and make batches _____________________________________

# Data folder
data_folder="../DATA/holo4k_preprocessed/holo4k_batched"
EMB_NAME=args.emb_name
FEAT_SIZES={"onehot":21,
            "bert":1024,
            "xlnet":1024}


# else:

TEST_MODEL=args.test_mode

if args.no_graph:
    no_graph="_no_graph"
else:
    no_graph=""

if TEST_MODEL:
    TRAIN_MODE="model_dev"
    TEST_MODE="test"
    MODES=[TRAIN_MODE,TEST_MODE]
    evaluate=False
else:
    TRAIN_MODE=f"{args.fold}train"
    VAL_MODE=f"{args.fold}val"
    # TRAIN_MODE="model_dev"
    # VAL_MODE="test_dissim"
    MODES=[TRAIN_MODE,VAL_MODE]



def get_graphs(mode):
    with zipfile.ZipFile(data_folder+f'/{EMB_NAME}_{mode}{no_graph}_batched_{args.batch_size}.zip') as thezip:
        return thezip.namelist()

GRAPHS={mode:get_graphs(mode) for mode in MODES}




def get_batch(k,mode):
    # print(f"{mode} batches : {a} - {b}")
    with zipfile.ZipFile(data_folder+f'/{EMB_NAME}_{mode}{no_graph}_batched_{args.batch_size}.zip') as thezip:
        graph=GRAPHS[mode][k]
        batch_graph=pickle.load(thezip.open(graph,"r"))

    if args.no_emb:
        batch_graph.ndata['feat']=torch.ones_like(batch_graph.ndata['feat'])
    

    batch_graph=batch_graph.to('cuda:0')
    features = batch_graph.ndata['feat'].to('cuda:0')
    labels = batch_graph.ndata['label'].to('cuda:0')
    w0=0
    w1=0
    if mode==TRAIN_MODE:
        n0=int((labels==0).sum())
        n1=int((labels==1).sum())
        w0=n1/(n1+n0)
        w1=n0/(n1+n0)

    return batch_graph,features,labels,w0,w1

# Get results

if TEST_MODEL:
    RESULTS={"Epoch":[],"Train MCC":[]}
else:
    RESULTS={"Epoch":[],"Train MCC":[],"Val MCC":[]}



# Train the model    

def train(model,metric_name):
    optimizer = torch.optim.AdamW(model.parameters(),lr=0.0003)
    
    for e in range(args.epochs+1):
        # evaluate=(e%args.evaluate_each==0)

        if evaluate:
            train_pred=torch.Tensor([]).to("cuda:0")
            train_labels=torch.Tensor([]).to("cuda:0")
        a_train=0
        model.train()
        for a_train in range(len(GRAPHS[TRAIN_MODE])):
            # load batch
            g_train,features,labels,w0,w1=get_batch(a_train,TRAIN_MODE)
            # Forward propagation
            logits = model(g_train, features)
            if evaluate:
                # Get predictions and labels for evaluation
                train_pred = torch.concat([train_pred,logits.argmax(1)]) 
                train_labels = torch.concat([train_labels,labels]) 
            # Compute loss
            # Note that you should only compute the losses of the nodes in the training set.
            weight=(torch.Tensor([w0,w1])).to('cuda:0')
            loss=nn.CrossEntropyLoss(weight)(logits.float(), labels.reshape(-1,).long())
            # Backward propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        
        random.shuffle(GRAPHS[TRAIN_MODE])
        
        if evaluate:
            model.eval()
        # Get evaluation metric       
            train_metric=get_metric(train_pred,train_labels,metric_name)
            RESULTS["Epoch"].append(e+1)
            RESULTS["Train MCC"].append(train_metric)
            if TEST_MODEL:
                print('In epoch {}, loss: {:.3f}, train {} : {:.3f}'.format(e, loss,metric_name,train_metric))
            else:
                val_pred=torch.tensor([]).to("cuda:0")
                val_labels=torch.tensor([]).to("cuda:0")
                for a_val in range(len(GRAPHS[VAL_MODE])):
                    # load batch
                    g_val,features,labels,_,_=get_batch(a_val,VAL_MODE)
                    # Forward propagation
                    logits = model(g_val, features)
                    # Get predictions and labels for evaluation
                    val_pred = torch.concat([val_pred,logits.argmax(1)]) 
                    val_labels = torch.concat([val_labels,labels])

                val_metric=get_metric(val_pred,val_labels,metric_name)
                RESULTS["Val MCC"].append(val_metric)
                print('In epoch {}, loss: {:.3f}, train {} : {:.3f} , val {} : {:.3f}'.format(
                e, loss,metric_name,train_metric,metric_name ,val_metric))
                
            
        


# Configure training
print("BEGIN TRAINING ")
n_feats=FEAT_SIZES[EMB_NAME]


layers=[n_feats]+args.layers+[2]

if args.no_emb:
    print("No Embedding !!")
else:
    print("Embedding : ",EMB_NAME)

print(f"CV Fold : {args.fold}")

if args.no_graph:
    print("No Graph information !!")

print("Layers : ",layers)
print("Dropout rate : ",args.dropout)
if args.batch_norm:
    print("Using Batch Normalization")
if args.residual:
    print("Using Residual connections")
print(f"Training for {args.epochs} Epochs ")
print("Batch size : ",args.batch_size)

print("Training model...")

model = GCN(layers).cuda()

# Train the model
train(model,metric_name="mcc")


# Test the model
if TEST_MODEL:
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

    print("Test MCC ",get_metric(test_pred,test_labels,"mcc"))

    if (not args.no_graph) and (not args.no_emb):
        torch.save(model.state_dict(),f"/app/output/{EMB_NAME}_model_final.pt")
    
    if (args.no_graph):
        torch.save(model.state_dict(),f"/app/output/{EMB_NAME}_no_graph.pt")





