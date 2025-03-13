import os
import glob
import pickle
import numpy as np
import pandas as pd  
import csv
from scipy.io import savemat
from scipy.io import loadmat
from scipy import stats
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from functools import reduce 
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import random
from itertools import combinations
from itertools import permutations
from numpy.linalg import inv
import argparse

# subject=798
# model_name_base_df='gpt2-xl'
# model_name_emb='mistral-7b'
# min_num_word=0
# production=1
# if production==1:
#     status='prod'
# else:
#     status='comp'

# import argparse


def parse_arguments():
    """Read commandline arguments

    Returns:
        args (Namespace): input as well as default arguments
    """

    parser = argparse.ArgumentParser()

    # main args
    parser.add_argument("--subject",  type=int, required=True) 
    parser.add_argument("--min_num_words", type=int, required=True)
    parser.add_argument("--model_name_emb", type=str, required=True)
    parser.add_argument("--model_name_base_df", type=str, required=True)

    args = parser.parse_args()
    return args



def load_label(filepath):

    with open(filepath, "rb") as f:
        full_labels = pickle.load(f)
        labels_df = pd.DataFrame(full_labels)
    
    return labels_df

def get_embeddings(args):   

    if args.model_name_emb=='mistral-7b':
        path_emb='/scratch/gpfs/ij9216/projects/code/247/247-embedding-dev/results/798/pickles/embeddings/gpt2-medium/sentence/intfloat-e5-mistral-7b-instruct/full/cnxt_0001/layer_00.pkl'

    elif args.model_name_emb=='gpt2-xl':
        path_emb1='/scratch/gpfs/kw1166/247/247-pickling/results/tfs/' 
        path_emb1=path_emb1+str(args.subject)+'/pickles/embeddings/'+args.model_name_emb+'/full/cnxt_1024/'
        emb_file='layer_48.pkl'    
        path_emb=path_emb1+emb_file
    
    df_emb=load_label(path_emb)
    
    return df_emb

def get_df(args):

    path_base_df='/scratch/gpfs/kw1166/247/247-pickling/results/tfs/'  
    path_base_df=path_base_df+str(args.subject)+'/pickles/embeddings/'+args.model_name_base_df+'/full/base_df.pkl'

    df=load_label(path_base_df)
    df_emb=get_embeddings(args)

    df['embeddings']=df_emb['embeddings']


    df=df.loc[(~df.adjusted_onset.isnull()) & (df.token_idx==0) & (df.is_nonword==False)]

    # df=df[df.num_words>args.min_num_word]

    # inserting last word_index
    df['last_word_index']=0
    sentence_id=(df.sentence_idx.values)
    a1=list(sentence_id)
    a2=list(sentence_id[1:])
    a2.append(1)
    a3=np.asarray(a2)-np.asarray(a1)
    id=[]
    for x in a3:
        if x==0:
            id.append(0)
        else:
            id.append(1)

    id=np.asarray(id)

    df['last_word_index']=id
    df2=df[['word','onset','offset','sentence','sentence_idx','conversation_name','production','speaker','embeddings','last_word_index','num_words']]  

    return df2

def save_pickle(args):

    df2=get_df(args)
    
    onsets=[]
    offsets=[]
    audio_onsets=[]
    audio_offsets=[]
    production=[]
    new_sentence=[]
    num_words=[]
    conversation_name=[]
    sentence_idx=[]
    embeddings=[]
    speaker=[]
    all_onsets=[]

    conv_names=np.unique(df2.conversation_name)

    # word_to_remove=set(np.unique(df2[df2.token_type=='tag'].word.values))

    fs=44100
    f=fs/512

    q=0
    for conv_name in conv_names:

        if (q%1==0):
            print(q)
        q=q+1

        df3=df2[df2.conversation_name==conv_name]
        sentence_ids=np.unique(df3.sentence_idx.values)

        for sentence_id in sentence_ids:
            
            df3=df2[df2.conversation_name==conv_name]
            df3=df3[df3.sentence_idx==sentence_id]      
            s=df3.sentence.values[0]
            
            onsets.append(df3.onset.values[0])
            offsets.append(df3[df3.last_word_index==1].offset.values[-1])
            new_sentence.append(s)
            num_words.append(df3[df3.last_word_index==1].num_words.values[0])
            conversation_name.append(conv_name)
            sentence_idx.append(sentence_id)
            
            embeddings.append(df3[df3.last_word_index==1].embeddings.values[-1])
            
            a1=round((df3[df3.last_word_index==1].onset.values[0])*f)
            a2=round((df3[df3.last_word_index==1].offset.values[-1])*f)
        
            audio_onsets.append(a1)
            audio_offsets.append(a2)
            
            production.append(df3[df3.last_word_index==1].production.values[0])
            speaker.append(df3[df3.last_word_index==1].speaker.values[0])

            del s
            
            # if df3.speaker.values[0]=='Patient':
            #     production.append(1)
            # else:
            #     production.append(0)
                

    duration=(np.asarray(offsets)-np.asarray(onsets))/512

    db = {}

    db['onsets']=onsets
    db['offsets']=offsets
    db['duration']=duration
    db['sentence']=new_sentence
    db['num_words']=num_words
    db['conversation_name']=conversation_name
    db['production']=production
    db['speaker']=speaker
    db['audio_onsets']=audio_onsets
    db['audio_offsets']=audio_offsets
    db['sentence_idx']=sentence_idx
    db['embeddings']=embeddings

    # db['embeddings'] =embeddings
    # db['all_onsets'] =all_onsets

    os.chdir(r"/scratch/gpfs/arnab/flexible_encoding/pickles_for_flex_encoding")
    filename='utterance_'+args.model_name_emb+'_'+str(args.subject)+'.pkl'
    with open(filename, "wb") as f:
        pickle.dump(db, f)

    return None 

def main():
    args = parse_arguments()
    save_pickle(args)
    return None
if __name__ == "__main__":
    main()
