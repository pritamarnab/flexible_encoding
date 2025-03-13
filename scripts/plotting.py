import os
import ast
import glob
import pickle
import numpy as np
import pandas as pd  
import csv
import string
from scipy.io import savemat
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from functools import reduce 
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import argparse
import torch
from torch import nn
from einops import rearrange
import torch.utils.data as data
from torch.nn import functional as F
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchsummary import summary
from scipy.interpolate import interp1d
from utils import load_label
import argparse


def parse_arguments():
    """Read commandline arguments

    Returns:
        args (Namespace): input as well as default arguments
    """

    parser = argparse.ArgumentParser()

    # main args
    parser.add_argument("--subject",  type=int, required=True) 
    parser.add_argument("--total_duration", type=str, required=True)
    parser.add_argument("--model_name_emb", type=str, required=True)
    parser.add_argument("--folder_name", type=str, required=True)

    args = parser.parse_args()
    return args

def create_plot(args):

    os.makedirs(os.getcwd()+'/results/plots/'+args.folder_name,exist_ok=True)

    

    plt_pdf_names=['correlation_across_lags_sorted_descending','mse_error_across_lags','min_loss_indices_across_sentences']

    for k in range(3):

        name=os.getcwd()+'/results/plots/'+args.folder_name+'/'+plt_pdf_names[k]+'_'+args.total_duration+'_durations.pdf'
        
        pdf_pages = PdfPages(name)


        # pdf_pages = PdfPages('min_loss_indices_across_sentences.pdf')
        # pdf_pages = PdfPages('correlation_across_lags_sorted_descending.pdf')
        # pdf_pages = PdfPages('mse_error_across_lags.pdf')


        l=np.arange(-1,0,0.1)
        # subject=798

        path='/scratch/gpfs/kw1166/247/247-pickling/results/tfs/'+str(args.subject)+'/pickles/'+str(args.subject)+'_electrode_names.pkl'
            
        df_name_match=load_label(path)

        for elec_id in np.arange(1,192):
                
            # filename='/scratch/gpfs/arnab/flexible_encoding/results/mat_files/elec_result_'+str(elec_id)+'.mat'
            # filename='/scratch/gpfs/arnab/flexible_encoding/results/mat_files/elec_result_'+str(elec_id)+'_max_lag_15.0.mat'
            filename=os.getcwd()+'/results/mat_files/'+args.folder_name+'/elec_result_'+str(elec_id)+'_max_lag_15.0.mat'
            
            data=loadmat(filename)
            corr=data['corr'][0]
            mse_error=data['mse_error']
            lags=data['lags'][0]
            indices=data['min_loss_indices'][0]
            duration=data['test_durations'][0]

            if args.total_duration=='short':
                index=np.where(duration<=2)
                mse_error=np.squeeze(mse_error[index,:])
                indices=indices[index]
                duration=duration[index]

            elif args.total_duration=='medium':
                index=np.where((duration>=2) & (duration<=10))
                
                mse_error=np.squeeze(mse_error[index,:])
                indices=indices[index]
                duration=duration[index]

            elif args.total_duration=='long':
                index=np.where(duration>=10)
                mse_error=np.squeeze(mse_error[index,:])
                indices=indices[index]
                duration=duration[index]

            elec_name=df_name_match[df_name_match['electrode_id']==elec_id].electrode_name.values

            mse_error_2=np.zeros((np.shape(mse_error)[0],len(l)))

            for kk in range(np.shape(mse_error)[0]):

                
                # index=np.where(((lags/(duration[k]*1000))<=0) & ((lags/(duration[k]*1000))>=-1.2))
                index=np.where((abs(lags/(duration[kk]*1000))<=0) <1.2)
                mse_error_new=(mse_error[kk,index])
                lags_new=(lags[index]/(duration[kk]*1000))
                lags_new[-2]=0.1
                lags_new[-1]=0.2
                
                interpolation_function = interp1d(lags_new, mse_error_new, kind='linear')
                mse_error_2[kk,:] = interpolation_function(l)
            
            mean_error=np.mean(mse_error_2,axis=0)
            indices2=np.asarray([lags[idx] for idx in indices])/1000
            indices2=[indices2[idx]/duration[idx] for idx in range(len(indices2))]

            title_name= ['Correlation_', 'MSE_Error_', 'Min_loss_indices_']
            # corr_name= 'Correlation_'+elec_name
            # error_name= 'MSE_Error_'+elec_name
            # index_name= 'Min_loss_indices_'+elec_name
            title=title_name[k]+elec_name

            # breakpoint()

            plt.figure()

            if k==0:
                plt.plot(corr)
                plt.ylabel('Corr')
            elif k==1:
                plt.plot(l[3:],mean_error[3:])
                plt.ylabel('Error')
            else:
                plt.hist(indices2, bins=20, label=f'duration_str(duration({elec_id}))')
                plt.xlabel('lag_percentage')
            plt.title(title)

            # plt.plot(corr)
            # plt.ylabel('Corr')
            # plt.plot(l[3:],mean_error[3:])
            # plt.ylabel('Error')
            # plt.hist(indices2, bins=100, label=f'duration_str(duration({elec_id}))')
            # # plt.title(index_name[0])
            # plt.title(title)
            
            pdf_pages.savefig()  # Save the current figure
            plt.close()
            

        # Close the PDF file
        pdf_pages.close()

def main():
    args = parse_arguments()
    create_plot(args)

if __name__ == "__main__":
    main()