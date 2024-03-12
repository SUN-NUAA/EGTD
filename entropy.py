# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 18:03:52 2023

@author: foolinsky
"""

import numpy as np
from scipy.integrate import quad 

v=0

# Integrand function
def int1(z):
    global v
    p=np.exp(((v-z)**2)*-0.5)
    p=np.mean(p)/np.sqrt(np.pi*2)
    if p < 1e-10:
        return 0
    return -p*np.log(p)

# caculate frequency entropy of a time series according eq.10
def entropy1(x,h=0.25):  
    l=x.shape[0]
    x1=x-np.mean(x)
    global v
    v=np.fft.fft(x1)
    vr=np.real(v)[:l//2+1]
    vi=np.imag(v)[1:(l+1)//2]
    v=np.concatenate((vr,vi))
    ma,mi=np.max(v),np.min(v)
    v=v*(np.log2(l)/(ma-mi)/h)
    ma,mi=np.max(v)+5,np.min(v)-5
    return quad(int1,mi,ma)[0]

# caculate frequency entropy of a group of time series collected by 
# several data sources according eq.10
def entropy2(x,h=0.25):
    l=x.shape[1]
    x1=x-np.repeat(np.expand_dims(np.mean(x,axis=1),axis=1),l,axis=1)
    v2=np.fft.fft(x1,axis=1)
    vr=np.real(v2)[:,:l//2+1]
    vi=np.imag(v2)[:,1:(l+1)//2]
    v2=np.concatenate((vr,vi),axis=1)
    ma,mi=np.max(v2),np.min(v2)
    v2=v2*(np.log2(l)/(ma-mi)/h)
    ma,mi=np.max(v2)+5,np.min(v2)-5
    res=np.zeros((x.shape[0],))
    global v
    for i in range(x.shape[0]):
        v=v2[i,:]
        res[i]=quad(int1,mi,ma)[0]
    return res
    
