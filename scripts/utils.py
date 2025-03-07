import os
import glob
import pickle
import numpy as np
import pandas as pd  
import csv
import string
from scipy.io import savemat
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from functools import reduce 
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import argparse
import torch
from torch import nn
from einops import rearrange
import torch.utils.data as data
from torch.nn import functional as F
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchsummary import summary


def load_pickle(file_path):
    
    pickle_file = open(file_path, "rb")
    objects = []

    i=0

    while True:
        # print('i',i)
        try:

            objects.append(pickle.load(pickle_file))

        except EOFError:

            break

    pickle_file.close()

    a=objects[0]
    
    return a

def load_label(filepath):

    with open(filepath, "rb") as f:
        full_labels = pickle.load(f)
        #labels_df = pd.DataFrame(full_labels["labels"])
        labels_df = pd.DataFrame(full_labels)

    # labels_df["audio_onset"] = ((labels_df.onset + 3000) / 512)
    # labels_df["audio_offset"] = ((labels_df.offset + 3000) / 512)

    # labels_df = labels_df.dropna(subset=["audio_onset", "audio_offset"])
    

    return labels_df

def create_result(epoch, predicted_output, actual_output,result):

    predicted_output=torch.squeeze(predicted_output,1)
    actual_output=torch.squeeze(actual_output,1)

    # mse_error = torch.min(torch.square(predicted_output-actual_output),-1)

    mse_error = (torch.square(predicted_output-actual_output))
    sorted, indices = torch.sort(mse_error)

    sorted_actual= [actual_output[i,indices[i,:]] for i in range(actual_output.shape[0])]
    sorted_actual=torch.stack(sorted_actual)

    mean_loss= torch.mean(mse_error, 0)
    mean_loss = [t.cpu().numpy() for t in mean_loss]
    mse_error = [t.cpu().numpy() for t in mse_error]

    min_loss_lag_position=indices[:,0]
    min_loss_lag_position = [t.cpu().numpy() for t in min_loss_lag_position]
    corr=[]

    for i in range(sorted_actual.shape[1]):
        corr.append(np.corrcoef(predicted_output[:,i],sorted_actual[:,i])[0,1])


    df1=pd.DataFrame([[epoch, corr,mean_loss, mse_error,min_loss_lag_position]], columns=['epoch', 'corr', 'mean_loss', 'mse_error','min_loss_lag_position'])                        
    result=pd.concat([result,df1], ignore_index=True)                      

    # breakpoint()

    # predicted=[]
    # actual=[]

    # for ii in range(actual_output.shape[0]):    
    #     actual.append(vlabels[ii, 0, mse_error.indices[ii]])
    #     predicted.append(output1[ii, 0, mse_error.indices[ii]])

    # b = [t.cpu().numpy() for t in predicted]
    # del predicted
    # predicted=np.asarray(b)
    # predicted=predicted.flatten()

    # del b
    # b = [t.cpu().numpy() for t in actual]
    # del actual
    # y=np.squeeze(np.asarray(b))

    # corr=(np.corrcoef(predicted,y)[0,1])

    return result, corr[0]

def train_one_epoch(epoch_index,args, model, trainloader, optimizer, loss_fn):
    running_loss = 0.
    last_loss = 0.
    
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for batch_idx, batch in enumerate(trainloader):
        # Every data instance is an input + label pair
        x,y=batch
    
        # Zero your gradients for every batch!
        optimizer.zero_grad()
    
        # Make predictions for this batch
        x = x.to(torch.float32)
        x = x.to(args.device)
        y = y.to(torch.float32)
        y = y.to(args.device)

        if args.use_second_network:

            [output1,output2]=model(args,x)

            loss = loss_fn(output1,output2, y) # Compute the loss and its gradients

        else:
            output1=model(args,x)

            loss = loss_fn(output1, y) # Compute the loss and its gradients
    
        
        
        loss.backward()
    
        # Adjust learning weights
        optimizer.step()
    
        # Gather data and report
        running_loss += loss.item()

    last_loss = running_loss / (batch_idx+1) # loss per batch
    print('epoch {} loss: {}'.format(epoch_index, last_loss))
            
    return last_loss

def training(args,model, trainloader, testloader, optimizer, loss_fn):

    corr=[]
    result = pd.DataFrame(columns=['epoch','corr','mean_loss', 'mse_error','min_loss_lag_position'])

    for epoch in range(args.EPOCHS):
        print('EPOCH {}:'.format(epoch + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch,args, model, trainloader, optimizer, loss_fn)
        

        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        predicted=[]
        actual=[]

        p=0

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(testloader):
                vinputs, vlabels = vdata
                vinputs = vinputs.to(torch.float32)
                vinputs = vinputs.to(args.device)
                vlabels = vlabels.to(torch.float32)
                vlabels = vlabels.to(args.device)

                if args.use_second_network:
                    [output1,output2]=model(args,vinputs)
                    vloss = loss_fn(output1,output2, vlabels)
                else:
                    output1=model(args,vinputs)
                    vloss = loss_fn(output1, vlabels)
                
                running_vloss += vloss

                if p==0:
                    predicted_output=output1
                    actual_output=vlabels
                    p=p+1

                else:
                    predicted_output=torch.cat((predicted_output,output1),dim=0)
                    actual_output=torch.cat((actual_output,vlabels),dim=0)

        [result, c]= create_result(epoch, predicted_output, actual_output,result)
        corr.append(c)
        print('Corr:',c)

    print('final_corr')
    print(np.max(corr))
    best_epoch=np.argmax(corr)

    epoch=result.loc[best_epoch,'epoch']
    corr=result.loc[best_epoch,'corr']
    mean_loss=result.loc[best_epoch,'mean_loss']
    mse_error=result.loc[best_epoch,'mse_error']
    indices=result.loc[best_epoch,'min_loss_lag_position']

    return corr, mean_loss, mse_error, indices