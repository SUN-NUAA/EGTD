# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 16:47:01 2023

@author: foolinsky
"""

import os
import pandas as pd
import numpy as np
import L
import entropy
import egtd
import warnings
warnings.filterwarnings("ignore")

setPath='D:\\Data\\UCR_TS_Archive_2015'
setNames = os.listdir(setPath)

m=4         #number of uncertain sources

nmin=1000   #if number of samples in a dataset less than nmin,
            #each sample will be calculated multiple times

print("DataSet", "RMSE", "MAE")

#iterate each sub-dataset
for setName in setNames:
    # Read a dataset from ucr archive
    df1=pd.read_csv(setPath+'\\'+setName+'\\'+setName+'_TEST',header=None)
    df2=pd.read_csv(setPath+'\\'+setName+'\\'+setName+'_TRAIN',header=None)
    df=pd.concat((df1,df2))
    # Create m uncertain source from each sample in the dataset df 
    samp=df.values[:,1:]
    del df,df1,df2
    lp=L.lprime(samp.shape[1])
    samp=samp[:,0:lp]
    samp=np.expand_dims(samp,axis=1)
    samp=np.repeat(samp,m+1,axis=1)
    samp=np.repeat(samp,(nmin+samp.shape[0]-1)//samp.shape[0],axis=0)

    #randomly generate std of each source
    std=np.abs(np.random.randn(samp.shape[0],m+1))
    std[:,0]=0
    err=std
    err=np.expand_dims(err,axis=2)
    err=np.repeat(err,samp.shape[2],axis=2)
    #generate err at each time stamp according to the corresponding std
    err*=np.random.randn(samp.shape[0],samp.shape[1],samp.shape[2])
    samp+=err
    
    #run egtd algrithm on each sample
    for j in range(samp.shape[0]):
        #print(std[j])
        #print(entropy.entropy2(samp[j]))
        w=egtd.opt(samp[j,1:,:])
        truth=samp[j,0,:]
        ftruth=np.dot(np.expand_dims(w,axis=0),samp[j,1:,:]).reshape((lp,))
        rmse=np.sqrt(np.mean((ftruth-truth)**2))
        mae=np.mean(np.abs(ftruth-truth))
        print(setName, rmse, mae)
        break