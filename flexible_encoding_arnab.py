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

def parse_arguments():
    """Read commandline arguments

    Returns:
        args (Namespace): input as well as default arguments
    """

    parser = argparse.ArgumentParser()

    # main args
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--subject",  type=int, required=True) 
    parser.add_argument("--hidden_layer_dim",  type=int, required=True)
    parser.add_argument("--hidden_layer_num",  type=int, required=True)
    parser.add_argument("--lags",  nargs="+", type=int, required=True)
    parser.add_argument("--EPOCHS", type=int, required=True)
    parser.add_argument("--train_num", type=int, required=True) 
    parser.add_argument("--min_num_words", type=int, required=True)
    parser.add_argument("--use_second_network", action="store_true")
    parser.add_argument("--electrode", type=int, required=True)
    parser.add_argument("--taking_words", type=bool, required=True) 
    parser.add_argument("--num_words", type=int, required=True)
    parser.add_argument("--activation_function", type=str, required=True)
    parser.add_argument("--learning_rate",  type=float, required=True) 
    parser.add_argument("--momentum",  type=float, required=True) 
    # parser.add_argument("--selected_elec_id", action="store_true") 
    # parser.add_argument("--across_subject_with_repacing_srm", action="store_true")
    
    # parser.add_argument("--data-dir", type=str, required=True)
    # parser.add_argument("--saving-dir", type=str, required=True)
    # parser.add_argument("--freeze-decoder", action="store_true")
    
    # parser.add_argument("--z_score", action="store_true")
    # parser.add_argument("--make_position_embedding_zero", action="store_true")

    args = parser.parse_args()
    return args


args = parse_arguments()

# hyperparameters

batch_size=args.batch_size
num_words=500
emb_dim=50
hidden_layer_dim=args.hidden_layer_dim
EPOCHS = args.EPOCHS
train_num=args.train_num
electrode=args.electrode  #which electrode are we considering
taking_words=args.taking_words  # are we taking the actual words or the lags around the last word
lags=args.lags # only valid if taking word is false and should equal to the lag_number

if args.taking_words:

    lag_number=args.num_words
else:
    lag_number=len(lags) # how many previous lags/words to consider
subject=args.subject
min_num_words=args.min_num_words 

print(args)

# breakpoint()

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

def get_elec_id(subject):

    path='/projects/HASSON/247/plotting/sig-elecs/20230510-tfs-sig-file/'
    
    sig_file=path+'tfs-sig-file-glove-'+ str(subject)+'-'+'comp'+'.csv'
    
    df_sig=pd.read_csv(sig_file)
    
    elecs=df_sig.electrode.values
    
    
    
    path='/scratch/gpfs/kw1166/247/247-pickling/results/tfs/'+str(subject)+'/pickles/'+str(subject)+'_electrode_names.pkl'
    
    pickle_file = open(path, "rb")
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
    
    df_name_match=pd.DataFrame(a)
    
    elec_id=[]
    for elec in elecs:
        elec_id.extend(df_name_match[df_name_match.electrode_name==elec].electrode_id.values)
                      
    return elec_id




def get_elec_data(subject, df, ecogs, conv_name, all_onsets, lags, elec_id, lag_number,taking_words=True):

    # path='/projects/HASSON/247/data/conversations-car/'+str(subject)+'/'
    # # conv_name=df[df.conversation_id==conv_id].conversation_name.values[0]
    # # q1=glob.glob("*_comp_Y.npy")
    # # q1=glob.glob("NY625_*") ## all the conversation
    # # # path_elec=path+'/'+q1[0]+'/preprocessed/'
    
    
    
    # # Y_data= np.zeros((len(lags),len(onsets),elec_num)) 



    # ecogs=[]
    # for k1 in elec_id:
    #     filename=path+'/'+conv_name+'/preprocessed/'+conv_name+'_electrode_preprocess_file_'+str(k1)+'.mat'

    #     if subject==798:
    #         filename=path+'/'+conv_name+'/preprocessed_allElec/'+conv_name+'_electrode_preprocess_file_'+str(k1)+'.mat'
        
    #     e=loadmat(filename)['p1st'].squeeze().astype(np.float32)
    #     ecogs.append(e)
    
    # ecogs = np.asarray(ecogs).T



    elec_num=len(elec_id)
    t=len(ecogs[:,0])   
    window_size=200
    half_window = round((window_size / 1000) * 512 / 2)

    if taking_words:

        Y_data= np.zeros((elec_num, lag_number))    

        onsets=all_onsets[-lag_number:-1]
        onsets=np.append(onsets,all_onsets[-1])
    else:

        onsets=[]
        Y_data= np.zeros((elec_num, len(lags)))    

        for i in lags:

            lag_amount = int(i/ 1000 * 512)

            onsets.append(np.minimum(
                t - half_window - 1,
                np.maximum(half_window + 1,
                            np.round(all_onsets[-1]) + lag_amount)))

    
    
    index_onsets=np.asarray(onsets)

   
   
    
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
            
                            
        #if subject==717:
           
        Y_data[k,:] = np.mean(Y1, axis=-1)

    return Y_data

def all_ecog(elec_id, conv_name,subject ):

    
    elec_id=get_elec_id(subject)
    
    path='/projects/HASSON/247/data/conversations-car/'+str(subject)+'/'
    # conv_name=df[df.conversation_id==conv_id].conversation_name.values[0]
    # q1=glob.glob("*_comp_Y.npy")
    # q1=glob.glob("NY625_*") ## all the conversation
    # # path_elec=path+'/'+q1[0]+'/preprocessed/'
    
    elec_num=len(elec_id)
    
    ecogs=[]
    for k1 in elec_id:
        filename=path+'/'+conv_name+'/preprocessed/'+conv_name+'_electrode_preprocess_file_'+str(k1)+'.mat'
        
        if subject==798:
            filename=path+'/'+conv_name+'/preprocessed_allElec/'+conv_name+'_electrode_preprocess_file_'+str(k1)+'.mat'
        
        e=loadmat(filename)['p1st'].squeeze().astype(np.float32)
        ecogs.append(e)
        
    ecogs = np.asarray(ecogs).T

    return ecogs



#Preparing Data

# df=pd.read_csv('/scratch/gpfs/arnab/247_data_read/decoding_df_798_corrected.csv')
df=pd.read_csv('/scratch/gpfs/arnab/247_data_read/decoding_df_798_final.csv')

data=load_pickle('/scratch/gpfs/arnab/247_data_read/last_word_embeddings.pkl')

emb=data['embeddings']
a=data['all_onsets']

df['embeddings']=emb
df['all_onsets']=a

df=df.drop_duplicates(subset=['onsets'])

df=df[df.num_words>min_num_words]
embeddings=df.embeddings.values
all_onsets=df.all_onsets.values

df=df[df.corrupted==0]

subject=args.subject

elec_id=get_elec_id(subject)

conv_names=np.unique(df.conversation_name.values)

if taking_words:
    electrode_data=np.zeros((len(df.conversation_name.values),len(elec_id),lag_number)) 
else:
    electrode_data=np.zeros((len(df.conversation_name.values),len(elec_id),len(lags))) 

embeddings=[]

print('preparing electrode data')

p=0
for conv_name in conv_names:

    print(p)

    ecogs=all_ecog(elec_id, conv_name,subject)

    df2=df[df.conversation_name==conv_name]

    emb=df2.embeddings.values
    all_onsets=df2.all_onsets.values

    # if p==0:
    #    embeddings=emb
    # else:
    #     embeddings=np.concatenate((embeddings,emb),axis=0)

    for k in range(len(all_onsets)): 

        # print(k)
    
        conv_name=df.conversation_name.values[k]
        x=all_onsets[k]

        if len(x)>= lag_number and x[-1] < len(ecogs[:,0]):
        
            a1= get_elec_data(subject, df, ecogs, conv_name, x, lags, elec_id, lag_number,taking_words)
            electrode_data[p,:,:]=a1
            embeddings.append(emb[k])
            p=p+1

    # p=p+1
electrode_data=electrode_data[:p,:,:]
embeddings=np.asarray(embeddings)
pca = PCA(n_components=emb_dim)
embeddings=pca.fit_transform(embeddings)

scaler = StandardScaler()
embeddings=scaler.fit_transform(embeddings)

embeddings = np.expand_dims(embeddings, axis=1)


print('embeddings shape:',np.shape(embeddings))
print('elec data shape:',np.shape(electrode_data))

elec_data=electrode_data[:,electrode,:]

scaler = StandardScaler()
elec_data=scaler.fit_transform(elec_data)

elec_data = np.expand_dims(elec_data, axis=1)


X_train=torch.from_numpy(embeddings[:train_num,:,:])
y_train=torch.from_numpy(elec_data[:train_num,:,:])

X_test=torch.from_numpy(embeddings[train_num:,:,:])
y_test=torch.from_numpy(elec_data[train_num:,:,:])

trainset = torch.utils.data.TensorDataset(X_train, y_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)

testset = torch.utils.data.TensorDataset(X_test, y_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, drop_last=True)



print('electrode data prep done')

#Model

class flex_encoding(nn.Module):
    def __init__(self, emb_dim, lag_number,hidden_layer_dim):
        super().__init__()
        # the 3 Linear layers of the MLP
        self.fc1 = nn.Linear(emb_dim, 1)
        self.fc2 = nn.Linear(emb_dim, hidden_layer_dim)
        self.fc3 = nn.Linear(hidden_layer_dim, hidden_layer_dim)
        self.fc4 = nn.Linear(hidden_layer_dim, 1)
        self.fc5 = nn.Linear(emb_dim, lag_number)
        # self.fc3 = nn.Linear(n_hidden, num_classes)

        if args.activation_function=='ReLU':
            self.activition=nn.ReLU()
    
    def forward(self, x1):
        
        if args.hidden_layer_num!=0:
            x= self.activition(self.fc2(x1))

            for k in range(args.hidden_layer_num):

                x= self.activition(self.fc3(x))
            
            x= (self.fc4(x))

        else:
            x= (self.fc1(x1))

        y_pred=torch.repeat_interleave(x, lag_number, dim=-1)

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


    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model=flex_encoding(emb_dim, lag_number,hidden_layer_dim)
model = model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)

if args.use_second_network:
    loss_fn = CustomLoss()
else:
    # loss_fn = nn.MSELoss()
    loss_fn = CustomLoss_min_MSE()

print('Model Summary:')
print(summary(model, (3, 1, 50)))

def train_one_epoch(epoch_index,args):
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
        x = x.to(device)
        y = y.to(torch.float32)
        y = y.to(device)

        if args.use_second_network:

            [output1,output2]=model(x)

            loss = loss_fn(output1,output2, y) # Compute the loss and its gradients

        else:
            output1=model(x)

            loss = loss_fn(output1, y) # Compute the loss and its gradients
    
        
        
        loss.backward()
    
        # Adjust learning weights
        optimizer.step()
    
        # Gather data and report
        running_loss += loss.item()

    last_loss = running_loss / (batch_idx+1) # loss per batch
    print('epoch {} loss: {}'.format(epoch_index, last_loss))
            
    return last_loss
        
        
    
#Training
corr=[]

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch,args)


    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    predicted=[]
    actual=[]

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(testloader):
            vinputs, vlabels = vdata
            vinputs = vinputs.to(torch.float32)
            vinputs = vinputs.to(device)
            vlabels = vlabels.to(torch.float32)
            vlabels = vlabels.to(device)

            if args.use_second_network:
                [output1,output2]=model(vinputs)
                vloss = loss_fn(output1,output2, vlabels)
            else:
                output1=model(vinputs)
                vloss = loss_fn(output1, vlabels)
            
            running_vloss += vloss

            a1=torch.min(output1,dim=-1)
            predicted.append((a1.values))

            for ii in range(vlabels.shape[0]):    
                actual.append(vlabels[ii, 0, a1.indices[ii]] )

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    b = [t.cpu().numpy() for t in predicted]
    del predicted
    predicted=np.asarray(b)
    predicted=predicted.flatten()

    del b
    b = [t.cpu().numpy() for t in actual]
    del actual
    y=np.squeeze(np.asarray(b)) 

    # breakpoint()
    
    
    corr.append(np.corrcoef(predicted,y)[0,1])
    print('Corr:',np.corrcoef(predicted,y)[0,1])
    del y

print('final_corr')
print(np.max(corr))

        
