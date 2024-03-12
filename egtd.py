# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 22:48:12 2023

@author: foolinsky
"""

import numpy as np
from scipy.integrate import quad
import entropy

i,j,v,vhat,ma,mi=0,0,0,0,0,0

# p according eq.15
def p(z):
    global vhat
    return np.mean(np.exp(((z-vhat)**2)*-0.5))/np.sqrt(np.pi*2)

# The partial of p with respect to w_i accoroding to eq.24 
def pw1(z):
    global v,vhat,i
    t=z-vhat
    return np.mean(np.exp((t**2)*-0.5)*v[i,]*t)/np.sqrt(np.pi*2)

# The partial of p with respect to w_j accoroding to eq.24 
def pw2(z):
    global v,vhat,j
    t=z-vhat
    return np.mean(np.exp((t**2)*-0.5)*v[j,]*t)/np.sqrt(np.pi*2)

# The second partial of z with respect to w_i and w_j according eq.25
def pww(z):
    global v,vhat,i,j
    t=z-vhat
    return np.mean(np.exp((t**2)*-0.5)*v[i,]*v[j,]*(t*t-1))/np.sqrt(np.pi*2)

# Integrand function of H according to eq.14
def hint(z):
    if p(z)<1e-10:
        return 0
    return -np.log(p(z))*p(z)   

# Integrand function of the partial of p with respect to w accoroding to eq.22 
def hwint(z):
    if p(z)<1e-10:
        return 0
    return -np.log(p(z))*pw1(z)   

# Integrand function of the second partial of p with respect to w accoroding to eq.23 
def hwwint(z):
    if p(z)<1e-8:
        return 0
    return -(np.log(p(z))*pww(z)+pw1(z)*pw2(z)/p(z)) 

#init fft parameters
def init(x,w,h=0.25):
    l=x.shape[1]
    x1=x-np.repeat(np.expand_dims(np.mean(x,axis=1),axis=1),l,axis=1)
    v2=np.fft.fft(x1,axis=1)
    vr=np.real(v2)[:,:l//2+1]
    vi=np.imag(v2)[:,1:(l+1)//2]
    v2=np.concatenate((vr,vi),axis=1) 
    global v,vhat,i,ma,mi
    ma,mi=np.max(v2),np.min(v2)
    v2=v2*(np.log2(l)/(ma-mi)/h)
    ma,mi=np.max(v2)+5,np.min(v2)-5
    v=v2
    vhat=np.sum(v2*np.repeat(np.expand_dims(w,axis=1),l,axis=1),axis=0)

# caculate H according eq.14
def H(x,w,h=0.25):
    l=x.shape[1]
    global ma,mi,vhat
    vhat=np.sum(v*np.repeat(np.expand_dims(w,axis=1),l,axis=1),axis=0)
    return quad(hint,mi,ma,epsrel=1e-5,limit=100)[0]

# The partial of p with respect to w accoroding to eq.22 
def Hw(x,w,h=0.25):
    global i,ma,mi
    res=np.zeros(w.shape)
    for k in range(w.shape[0]):
        i=k      
        res[k]=quad(hwint,mi,ma,epsrel=1e-5,limit=100)[0]
    return res

# The second partial of p with respect to w accoroding to eq.23 
def Hww(x,w,h=0.25):
    global i,j,ma,mi
    res=np.zeros((w.shape[0],w.shape[0]))
    for k in range(w.shape[0]):
        for m in range(w.shape[0]):
            i,j=k,m      
            res[k,m]=quad(hwwint,mi,ma,epsrel=1e-5,limit=100)[0]
    return res

# Calulate dw according eqs.26 and 31
def gradient(x,w,h=0.25,lamb=1e-5):    
    m=w.shape[0]
    a,b=np.zeros((m+1,m+1)),np.zeros((m+1))
    a[0:m,0:m]=Hww(x,w,h)+np.identity(m)*lamb/w/w                
    a[m,0:m]=1
    a[0:m,m]=1
    b[0:m]=-Hw(x,w,h)+lamb/w
    dw=np.linalg.solve(a,b)[:m]
    s0=min(0.5/np.max(-dw/w),1)
    return (dw*s0,b[0:m])

# Newton opt algrithm
def opt(x,h=0.25,lamb=1e-5,rtol=1e-4,limit=100,sigma=0.9,gamma=0.45):
    ent=entropy.entropy2(x,h)
    w=np.exp(-ent)
    w/=w.sum()
    global v
    for k in range(limit):
        init(x,w,h)
        dw,g=gradient(x, w)
        for kk in range(limit):
            w1=w+dw
            en=H(x,w)
            en1=H(x,w1)
            if en1<=en+gamma*np.dot(g,dw):
                break
            dw*=sigma
        w+=dw
        if np.sum(np.abs(dw))<rtol:
            break
    return w

    


