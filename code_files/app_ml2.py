# ADDRESS VARIABLE
"""
IF THE ADDRESS SENT BY INTERFACE MATCHES THIS ADDRESS THEN PROCEED - DOWNLOAD THE WEIGHTS FROM
THE BLOCKCHAIN
UPDATE THE CURRENT WEIGHTS AS DOWNLOADED WEIGHTS
TRAIN
CHECK ACCURACY
IF ACCURACY IS GRATER, THEN SEND THE WEIGHTS, FLAG AND ACCURACY TO THE INTERFACE

"""

address = '0xAB0313359bAc4278bA5873c355EB5D20356D7238'
# XOR code
#
# We need this ML code here because it would be available locally on all nodes
from collections import Counter
from collections import defaultdict
from random import random
from collections import defaultdict
from IPython import display
from PIL import Image
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn.functional as F 
from torchvision import models
import json
import itertools
import torch
import torch.nn as nn
import torch.optim as optim 
import matplotlib.ticker as ticker 
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import json
import math
from web3 import Web3
class simpleNN(nn.Sequential):
    def __init__(self, input_size, hidden_size, output_size):
        super(simpleNN, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.l2 = nn.Linear(hidden_size, output_size)
    def foward(self, x):
        out = self.l1(x)
        out = self.sigmoid(out)
        out = self.l2(out)
        out = self.sigmoid(out)
        return out

# generating a random data set
#Xr= np.random.randint(100, size=(10000,2))
#Xr = np.square(Xr)
inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
expected_output = np.array([[0],[1],[1],[0]])
#Yr = Xr.sum(axis = 1)
#Yr = Yr.reshape(Yr.shape[0],1)
# creat  dataset 
class randomDataset(Dataset):
    def __init__(self,X,Y):
        self.n_samples = X.shape[0]
        self.X = torch.from_numpy(X)
        self.X = self.X.float()
        #print (self.X)
        self.Y = torch.from_numpy(Y)
        self.Y = self.Y.float()
    def __getitem__(self,index):
        return self.X[index], self.Y[index]
    
    def __len__(self):
        return self.n_samples
# set variables
ninputs = 2
nhidden = 5
noutputs = 1 # here l2 will be a single dimensuinal array
model1 = simpleNN(ninputs, nhidden, noutputs)
learning_rate=0.01
# loss and optimizer
#criterion = nn.MultiLabelSoftMarginLoss()
criterionr = nn.MSELoss()
optimizerr = torch.optim.SGD(model1.parameters(), lr = learning_rate )
datasetr = randomDataset(inputs,expected_output)
dataloader_trainr = DataLoader(dataset = datasetr, shuffle = True)
dataiter = iter(dataloader_trainr)
# train
tl = []
last_accuracy = 0
def train2(model, learning_rate=0.01, epochs=100):
    """
    Training function which takes as input a model, a learning rate and a batch size.
  
    After completing a full pass over the data, the function exists, and the input model will be trained.
    """
    # -- Your code goes here --
    for i in range(0,epochs):
        model.train()
        for  bindex, (featueres, labels) in enumerate(dataloader_trainr):
            #print (featueres.shape)
            #print (b_index)
            outputs = model(featueres)
            loss = criterionr(outputs,labels)
            loss.backward()
            optimizerr.step()
            optimizerr.zero_grad()
            tl.append(loss)
# evaluate the model

def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        # round to class values
        yhat = yhat.round()
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = np.vstack(predictions), np.vstack(actuals)
    # calculate accuracy
    acc = accuracy_score(actuals, predictions)
    return acc
#train2(model1)
test_inputs = np.array([[0,0],[0,1],[1,0],[1,1], [0,0],[0,1],[1,0],[1,1]])
expected_test_output = np.array([[0],[1],[1],[0], [0],[1],[1],[0] ])
dataset_test = randomDataset(test_inputs,expected_test_output)
dataloader_test = DataLoader(dataset = dataset_test, shuffle = True)
dataiter_test = iter(dataloader_test)
#ans = evaluate_model(dataloader_test, model1)
# for now taking the easiest accuracy -
"""
add: account that will carry out transactions, will also be the default account here
"""
def play(web3,add, first, account_index, contract):
    print ("##############################################################")
    if add == address:
        # call all functions
        web3.eth.defaultAccount = web3.eth.accounts[account_index]  
        if first ==1:
            #1. if this is the very first time the progrsm is being run - then simply run the module and
            # update the weights and flag variables in the smart contract
            # create proper signed transaction
            print ("First run")
            first = 0
            train2(model1)
            accuracy =evaluate_model(dataloader_test, model1)
            accuracy = math.floor(accuracy*1000000)
            weights1 = (model1.l1.weight)*1000000 # we need to convert the weights to uint
            weights2 = (model1.l2.weight)*1000000
            print (weights1)
            #print ("Shape of weights")
            #print (weights1.shape)
            shape1_w1 = weights1.shape[0]
            shape2_w1 = weights1.shape[1]
            shape1_w2 = weights2.shape[0]
            shape2_w2 = weights2.shape[1]
            # update the weights in the blockchain
            # updating the first weight matrix
            # we need to convert the weights into integers first
            for j in range (shape1_w1):
                for k in range (shape2_w1):
                    #val = model1.l1.weight[j][k]
                    val = weights1[j][k]
                    tx_hash = contract.functions.store2darray1(j,k,math.floor(val)).transact()
                    tx_receipt = web3.eth.waitForTransactionReceipt(tx_hash)
            print ("Updated weights in the blockchain for layer 1")
            print (contract.functions.getwodarray1().call())

            # updating the second weight matrix
            for j in range (shape1_w2):
                for k in range (shape2_w2):
                    #val = model1.l2.weight[j][k]
                    val = weights2[j][k]
                    tx_hash = contract.functions.store2darray2(j,k,math.floor(val)).transact()
                    tx_receipt = web3.eth.waitForTransactionReceipt(tx_hash)
            print ("Updated weights in the blockchain for layer 2")
            print (contract.functions.getwodarray2().call())
            # update the flag variable - is it required?
            # update the accuracy in the blockchain
            tx_hash = contract.functions.setAccuracy(accuracy).transact()
            tx_receipt = web3.eth.waitForTransactionReceipt(tx_hash)
            last_accuracy = accuracy
            

        else:
            
            #2.
            #for param in model1.parameters(): # needs to be done to preent leaf variable issue - we are manually updating weights
            #    param.requires_grad = False
            current_accuracy = contract.functions.getAccuracy().call()
            print("second run")
            current_accuracy = current_accuracy/1000000
            # download weights, train, then comare accuracy, if increased then update variables
            # download the weights, train the model and then update only if new accuracy is higher
            w1 = contract.functions.getwodarray1().call()
            # w1 is a list - it needs to be converted to appropriate type
            #Also - reshape the matrixes - oppposite sizes
            w1 = np.array(w1)
            print ("Stored weights downloaded from the blockchain")
            print (w1)
            #print (w1)
            '''
            for m in range (w1.shape[0]):
                for n in range (w1.shape[1]):
                    print (m,n, w1[m][n])
                print ()
            '''
            #print (type(w1))
            #print (w1.shape)
            w1 = w1/1000000
            w1 = torch.from_numpy(w1)
            #print (w1)
            w2 = contract.functions.getwodarray2().call()
            w2 = np.array(w2)
            '''
            for m in range (w2.shape[0]):
                for n in range (w2.shape[1]):
                    print (m,n, w2[m][n])
                print ()

            '''
            print (w2)
            w2 = w2/1000000
            w2 = torch.from_numpy(w2)
            #model1.l1.weight = w1
            print ("Model weights layer 1 that were actually initialized by default")
            print (model1.l1.weight)
            print ("Model weights layer 2 that were actually initialized by default")
            print (model1.l2.weight)
            #model1.l2.weight = w2
            # update the weights
            '''
            for m in range (w1.shape[0]):
                for n in range (w1.shape[1]):
                    model1.l1.weight[m][n] = w1[m][n]
                    #print (m,n, w1[m][n])
            for m in range (w2.shape[0]):
                for n in range (w2.shape[1]):
                    model1.l2.weight[m][n] = w2[m][n]
                    #print (m,n, w2[m][n])
            '''
            with torch.no_grad():
                model1.l1.weight = nn.Parameter(w1.float())
                model1.l2.weight = nn.Parameter(w2.float())
            
            print ("Training the model")
            train2(model1)
            print ("Updated weights after training of second model")
            print (model1.l1.weight)
            print ("------------------")
            print (model1.l2.weight)
            print ("------------------")
            accuracy =evaluate_model(dataloader_test, model1)
            print ("ACCURACY:", accuracy)
            # If new accuracy is greater then update the weights and accuracy
            if accuracy>= current_accuracy:
                accuracy = math.floor(accuracy*1000000)
                weights1 = (model1.l1.weight)*1000000
                weights2 = (model1.l2.weight)*1000000
                shape1_w1 = weights1.shape[0]
                shape2_w1 = weights1.shape[1]
                shape1_w2 = weights2.shape[0]
                shape2_w2 = weights2.shape[1]
                # update the weights in the blockchain
                # updating the first weight matrix
                # we need to convert the weights into integers first
                for j in range (shape1_w1):
                    for k in range (shape2_w1):
                        #val = model1.l1.weight[j][k]
                        val = weights1[j][k]
                        tx_hash = contract.functions.store2darray1(j,k,math.floor(val)).transact()
                        tx_receipt = web3.eth.waitForTransactionReceipt(tx_hash)
                print ("Updating the weights in blockchain for layer 1")
                print (contract.functions.getwodarray1().call())

                # updating the second weight matrix
                for j in range (shape1_w2):
                    for k in range (shape2_w2):
                        #val = model1.l2.weight[j][k]
                        val = weights2[j][k]
                        tx_hash = contract.functions.store2darray2(j,k,math.floor(val)).transact()
                        tx_receipt = web3.eth.waitForTransactionReceipt(tx_hash)
                print ("Updating the weights in blockchain for layer 2")
                print (contract.functions.getwodarray2().call())
                # update the flag variable - is it required?
                # update the accuracy in the blockchain
                tx_hash = contract.functions.setAccuracy(accuracy).transact()
                tx_receipt = web3.eth.waitForTransactionReceipt(tx_hash)
                last_accuracy = accuracy
            
            # create proper signed transaction
            
    else:
        pass
        # just pass - don't do anything

    #first is updated if this is the first time this entire program was run, else no change
    return first
