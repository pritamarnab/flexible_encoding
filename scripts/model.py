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


class flex_encoding(nn.Module):
    def __init__(self, args):
        super().__init__()
        # the 3 Linear layers of the MLP
        self.fc1 = nn.Linear(args.emb_dim, 1)
        self.fc2 = nn.Linear(args.emb_dim, args.hidden_layer_dim)
        self.fc3 = nn.Linear(args.hidden_layer_dim, args.hidden_layer_dim)
        self.fc4 = nn.Linear(args.hidden_layer_dim, 1)
        self.fc5 = nn.Linear(args.emb_dim, args.lag_number)
        # self.fc3 = nn.Linear(n_hidden, num_classes)

        if args.activation_function=='ReLU':
            self.activition=nn.ReLU()
    
    def forward(self, args, x1):
        
        if args.hidden_layer_num!=0:
            x= self.activition(self.fc2(x1))

            for k in range(args.hidden_layer_num-1):

                x= self.activition(self.fc3(x))
            
            x= (self.fc4(x))

        else:
            x= (self.fc1(x1))

        y_pred=torch.repeat_interleave(x, args.lag_number, dim=-1)

        if args.use_second_network:

            v= nn.Softmax(dim=1)(self.fc5(x1))

            return y_pred,v
        
        else:
            return y_pred
        
def bdot(a, b):  ##
    B = a.shape[0]
    S = a.shape[2]
    return torch.bmm(a.view(B, 1, S), b.view(B, S, 1)).reshape(-1)

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, y_pred, v, targets):
        mse_error = torch.square(targets-y_pred)
        loss=bdot(v, mse_error)
        return loss.mean()  

class CustomLoss_min_MSE(nn.Module):  

    #instead of MSE it takes the min loss across the lags for each sentence

    def __init__(self):
        super(CustomLoss_min_MSE, self).__init__()

    def forward(self, y_pred, targets):
        mse_error = torch.min(torch.square(targets-y_pred),-1).values
        
        return mse_error.mean()  