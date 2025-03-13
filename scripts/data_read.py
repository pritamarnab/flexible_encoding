import pickle
import numpy as np
import pandas as pd  
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

from config import parse_arguments
from utils import load_pickle, load_label


def get_elec_id(args):

    path='/projects/HASSON/247/plotting/sig-elecs/20230510-tfs-sig-file/'
    
    sig_file=path+'tfs-sig-file-glove-'+ str(args.subject)+'-'+'comp'+'.csv'
    
    df_sig=pd.read_csv(sig_file)
    
    elecs=df_sig.electrode.values
       
    path='/scratch/gpfs/kw1166/247/247-pickling/results/tfs/'+str(args.subject)+'/pickles/'+str(args.subject)+'_electrode_names.pkl'
    
    pickle_file = open(path, "rb")
    objects = []
    
     
    while True:
  
        try:
    
            objects.append(pickle.load(pickle_file))
    
        except EOFError:
    
            break
    
    pickle_file.close()
    
    a=objects[0]
    
    df_name_match=pd.DataFrame(a)
    
    elec_id=[]
    for elec in elecs:
        elec_id.extend(df_name_match[df_name_match.electrode_name==elec].electrode_id.values)

    if args.all_elec:
        elec_id=df_name_match.electrode_id.values.tolist()

        elec_id=[k for k in elec_id if k<193]
                      
    return elec_id

def get_elec_data(args,ecogs, all_onsets, offset, actual_onset, elec_id, ):

    high_value=args.HIGH_Value
    elec_num=len(elec_id)
    lags=args.lags
    t=len(ecogs[:,0])  # calucalting the length of the ecog signal   
    window_size=200
    half_window = round((window_size / 1000) * 512 / 2)

    if args.taking_words:

        lag_number=args.num_words
        Y_data= np.zeros((elec_num, lag_number))    

        onsets=all_onsets[-lag_number:-1]
        onsets=np.append(onsets,all_onsets[-1])
    else:

        lag_number=len(lags)
        onsets=[]
        Y_data= np.zeros((elec_num, len(lags)))    

        for i in lags:

            lag_amount = int(i/ 1000 * 512)

            onsets.append(np.minimum(
                t - half_window - 1,
                np.maximum(half_window + 1,
                            np.round(offset) + lag_amount)))

    index_onsets=np.asarray(onsets)

    
    # calculates the number of lags going past the sentence onset
    len_to_pad_high_value= np.shape(np.where(index_onsets < actual_onset))[1]

    
    for k in range(np.shape(ecogs)[1]):

        Y1 = np.zeros((len(onsets), 2 * half_window + 1))
        brain_signal=ecogs[:,k]
        
        # subtracting 1 from starts to account for 0-indexing
        starts = index_onsets - half_window - 1
        stops = index_onsets + half_window

        # print(starts)
        # print(stops)
        
        for i, (start, stop) in enumerate(zip(starts, stops)):
            start=int(start)
            stop=int(stop)
            Y1[i, :] = brain_signal[start:stop].reshape(-1)
    
           
        Y_data[k,:] = np.mean(Y1, axis=-1)

    Y_data[:, :len_to_pad_high_value]= high_value

    return Y_data

def all_ecog(elec_id, conv_name, args):

    path='/projects/HASSON/247/data/conversations-car/'+str(args.subject)+'/'

    ecogs=[]
    for k1 in elec_id:
        filename=path+'/'+conv_name+'/preprocessed/'+conv_name+'_electrode_preprocess_file_'+str(k1)+'.mat'
        
        if args.subject==798:
            filename=path+'/'+conv_name+'/preprocessed_allElec/'+conv_name+'_electrode_preprocess_file_'+str(k1)+'.mat'
        
        e=loadmat(filename)['p1st'].squeeze().astype(np.float32)
        ecogs.append(e)
        
    ecogs = np.asarray(ecogs).T

    return ecogs

def getting_df(args):

    if args.analysis_level=='sentence':

        # df=pd.read_csv('/scratch/gpfs/arnab/247_data_read/decoding_df_798_corrected.csv')
        df=pd.read_csv('/scratch/gpfs/arnab/247_data_read/decoding_df_798_final.csv')

        data=load_pickle('/scratch/gpfs/arnab/247_data_read/last_word_embeddings.pkl')

        emb=data['embeddings']
        a=data['all_onsets']

        df['embeddings']=emb
        df['all_onsets']=a

        df=df.drop_duplicates(subset=['onsets'])

        df['duration']=round((df.offsets-df.onsets)/512)

        df=df[df.num_words>args.min_num_words]
        # embeddings=df.embeddings.values
        # all_onsets=df.all_onsets.values
        
        df=df[df.corrupted==0]
        df=df[df.duration<args.max_duration]
    

    elif args.analysis_level=='utterance':

        if args.model_name_emb=='gpt2-xl':

            path_data='/scratch/gpfs/arnab/flexible_encoding/pickles_for_flex_encoding/utterance_gpt2-xl_798.pkl'

        elif args.model_name_emb=='mistral-7b':
            path_data='/scratch/gpfs/arnab/flexible_encoding/pickles_for_flex_encoding/utterance_mistral-7b_798.pkl'


        df=load_label(path_data)
        df=df[df.num_words>args.min_num_words]
        df=df[df.duration<args.max_duration]


    return df 

def custom_Zscore(x,args):
 

    for k in range(np.shape(x)[1]):
        
        y= x[:,k]
        y=y[y!=args.HIGH_Value]

        if len(y)>0:    
            x[:,k]=(x[:,k] - np.mean(y))/np.std(y)

    return x



def preparing_elec_data(args):

    df=getting_df(args)

    subject=args.subject

    # elec_id=get_elec_id(subject,args)
    elec_id=[args.electrode]

    conv_names=np.unique(df.conversation_name.values)

    if args.taking_words:
        lag_number=args.num_words
        electrode_data=np.zeros((len(df.conversation_name.values),len(elec_id),lag_number)) 
    else:
        
        electrode_data=np.zeros((len(df.conversation_name.values),len(elec_id),len(args.lags))) 

    embeddings=[]

    print('preparing electrode data')

    p=0
    durations=[]
    for conv_name in conv_names:

        print(conv_name)

        print(p)

        ecogs=all_ecog(elec_id, conv_name, args)


        df2=df[df.conversation_name==conv_name]

        emb=df2.embeddings.values
        if args.analysis_level=='sentence':
            all_onsets=df2.all_onsets.values

        offsets=df2.offsets.values
        actual_onsets=df2.onsets.values

        
        d=(df2.duration.values)

        # if p==0:
        #    embeddings=emb
        # else:
        #     embeddings=np.concatenate((embeddings,emb),axis=0)

        for k in range(len(offsets)): 

            # print(k)
        
            conv_name=df.conversation_name.values[k]
            x=0
            if args.analysis_level=='sentence':
                x=all_onsets[k]

            if offsets[k] < len(ecogs[:,0]):
        
                # a1= get_elec_data(subject, df, ecogs, conv_name, x, offsets[k], actual_onsets[k], args.lags, elec_id, lag_number,args.taking_words)
                a1=get_elec_data(args,ecogs, x, offsets[k], actual_onsets[k], elec_id, )
                electrode_data[p,:,:]=a1
                embeddings.append(emb[k])            
                durations.append(d[k])
                p=p+1

    electrode_data=electrode_data[:p,:,:]
    embeddings=np.asarray(embeddings)
    pca = PCA(n_components=args.emb_dim)
    embeddings=pca.fit_transform(embeddings)

    scaler = StandardScaler()
    embeddings=scaler.fit_transform(embeddings)

    embeddings = np.expand_dims(embeddings, axis=1)


    print('embeddings shape:',np.shape(embeddings))
    print('elec data shape:',np.shape(electrode_data))

    # elec_data=electrode_data[:,electrode,:]
    elec_data=electrode_data[:,0,:]

    # scaler = StandardScaler()
    # elec_data=scaler.fit_transform(elec_data)

    # breakpoint()

    elec_data=custom_Zscore(elec_data,args)

    elec_data = np.expand_dims(elec_data, axis=1)

    train_num=round(args.train_num*np.shape(elec_data)[0])

    X_train=torch.from_numpy(embeddings[:train_num,:,:])
    y_train=torch.from_numpy(elec_data[:train_num,:,:])

    X_test=torch.from_numpy(embeddings[train_num:,:,:])
    y_test=torch.from_numpy(elec_data[train_num:,:,:])

    durations=np.asarray(durations)
    test_durations=durations[train_num:]

    trainset = torch.utils.data.TensorDataset(X_train, y_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    testset = torch.utils.data.TensorDataset(X_test, y_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    print('electrode data prep done')

    return trainloader, testloader, test_durations



    