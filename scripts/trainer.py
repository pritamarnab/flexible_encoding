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


from utils import *
from model import *
from data_read import *
from config import *

def main():
    args = parse_arguments()
    if args.taking_words:
        args.lag_number=args.num_words
    else:
        args.lag_number=len(args.lags)

    print(args)

    breakpoint()

    # Load data
    [trainloader, testloader, test_durations] = preparing_elec_data(args)

    # Create model
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=flex_encoding(args)
    model = model.to(args.device)
    
    # Create optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
    # Create loss function
    if args.use_second_network:
        loss_fn = CustomLoss()
    else:
        # loss_fn = nn.MSELoss()
        loss_fn = CustomLoss_min_MSE()

    print('Model Summary:')
    # Train model

    #Training
    [corr, mean_loss, mse_error, indices] = training(args,model, trainloader, testloader, optimizer, loss_fn)

    # Save model
    # filename='/scratch/gpfs/arnab/flexible_encoding/results/mat_files/'+'elec_result_'+str(args.electrode)+'_max_lag_'+str(np.abs(lags[0]/1000))+'.mat'
    filename=args.save_dir+'elec_result_'+str(args.electrode)+'_max_lag_'+str(np.abs(args.lags[0]/1000))+'.mat'
    # df2.to_csv(filename)
  
    # savemat(filename,{'corr':corr,'mse_error':mse_error,'min_loss_indices':indices,'lags':args.lags, 'test_durations':test_durations})

    return None 

if __name__ == "__main__":
    main()