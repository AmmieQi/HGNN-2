#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 17:15:55 2018

@author: sulem
"""

import torch
import numpy as np
from random import shuffle

from utils import *
from preprocessing import graph_operators

# Check if CUDA is enabled
if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dtype_l = torch.cuda.LongTensor
    
else:
    dtype = torch.FloatTensor
    dtype_l = torch.LongTensor


def _divide_batch(nb_samples_in, batch_size, idx):
    """Returns a list of lists of indices of each mini batch given idx"""
    
    nb_batches = nb_samples_in // batch_size
    idx_list = []
    for i in range(0,nb_batches*batch_size, batch_size):
        idx_list.append(idx[i:i+batch_size])
    
    return idx_list


def _get_unsorted_batches(nb_samples_in, batch_size, shuffle_batch=False):
    """Create a list of indices idx then calls divide_batch"""
    idx = list(range(nb_samples_in))
    if (shuffle_batch==True):
        shuffle(idx)
    
    return _divide_batch(nb_samples_in, batch_size, idx)


def get_batches(nb_samples_in,batch_size,data,shuffle_batch=False,sort_batch=False):
    '''
    Gets batches indexes for a set of data
    For testing sets, grouping together batches of similar size allows speedup,
    where sample order doesn't matter.
    '''
    if sort_batch == False:
        return _get_unsorted_batches(nb_samples_in, batch_size, shuffle_batch)

    # Sort samples by nb_nodes
    sample_sizes = np.zeros(nb_samples_in)
    for i in range(len(data)):
        sample_sizes[i] = data[i][0].shape[0]
    sm_to_lg = np.argsort(sample_sizes)

    # Get batches
    idx_list = _divide_batch(nb_samples_in, batch_size, sm_to_lg)

    # Optionally shuffle order of batches
    if shuffle_batch == True:
        shuffle(idx_list)
    
    return idx_list
    

def prepare_batch(batch, task, J=1):
    
    """A batch will be a list of bs [x,A,t] instances
    
        x -> (N, n_features)
        A -> (N, N)
        t -> (n_tasks)
    
        Returns 7 tensors :
            
            X -> (bs , n_features , Nmax)
            W -> (bs , Nmax , Nmax , J)
            T -> (bs    )
            XL -> (bs , e_features , Emax)
            WL -> (bs , Emax , Emax , J)
            Pm -> (bs , Nmax, Emax)
            Pm -> (bs , Nmax, Emax)    """
    
    bs = len(batch)
    n_features = batch[0][0].shape[1]
    #n_tasks = batch[0][2].shape[0]

    N_batch = torch.zeros(bs).type(dtype_l)
    E_batch = torch.zeros(bs).type(dtype_l)
    for i in range(bs):
        N_batch[i] = batch[i][0].shape[0]
        E_batch[i] = (batch[i][1].nonzero()).shape[0]
    
    Nmax = torch.max(N_batch).item()
    Emax = torch.max(E_batch).item()
    
    #print("Nmax : {} Emax : {}".format(Nmax,Emax))
    
    npad_sizes = Nmax - N_batch
    epad_sizes = Emax - E_batch 
    
    mask = torch.zeros(bs, Nmax, Nmax)
    mask_lg = torch.zeros(bs, Emax, Emax)
    
    X = torch.zeros(bs, n_features, Nmax)
    W = torch.zeros(bs , Nmax , Nmax , J+2)
    T = torch.zeros(bs)
    XL = torch.zeros(bs , 1, Emax)
    WL = torch.zeros(bs , Emax , Emax , J+2)
    Pm = torch.zeros(bs , Nmax, Emax)
    Pd = torch.zeros(bs , Nmax, Emax)   
    
    # Pad samples with zeros
    for i in range(bs):
        
        #print("Molecule " + str(i))
        x, A, t = batch[i]
        
        #print("dual size : " + str(E_batch[i]))
        if npad_sizes[i] > 0 :
        
            z1 = torch.zeros(npad_sizes[i].item(), n_features)
            z2 = torch.zeros(N_batch[i].item(), npad_sizes[i].item())
            z3 = torch.zeros(npad_sizes[i].item(), Nmax)
            
            x = torch.cat((x, z1),dim=0)
            A = torch.cat(((torch.cat((A, z2),dim=1)),z3),dim=0) 
            
        w, wl, pm, pd = graph_operators([x,A],J,True)
        
        #print("w dimensions : " + str(w.shape))
        #print("wl dimensions : " + str(wl.shape))
        #print("pm dimensions : " + str(pm.shape))
        #print("pd dimensions : " + str(pd.shape))
        
        if epad_sizes[i] > 0 :
        
            z4 = torch.zeros(E_batch[i].item(), epad_sizes[i].item(), J+2)
            z5 = torch.zeros(epad_sizes[i].item(), Emax, J+2)
            z6 = torch.zeros(Nmax, epad_sizes[i].item())
            
            wl = torch.cat((torch.cat((wl, z4),dim=1),z5),dim=0)
            pm = torch.cat((pm, z6),dim=1)
            pd = torch.cat((pd, z6),dim=1)
        
        #print("w dimensions : " + str(w.shape))
        #print("wl dimensions : " + str(wl.shape))
        #print("pm dimensions : " + str(pm.shape))
        #print("pd dimensions : " + str(pd.shape))
        xl = torch.diag(wl[:,:,1])
        x = x.transpose(1,0)
        
        X[i,:,:].copy_(x)
        T[i] = t[task]
        XL[i,:,:].copy_(xl)
        W[i,:,:,:].copy_(w)
        WL[i,:,:,:].copy_(wl)
        Pm[i,:,:].copy_(pm)
        Pd[i,:,:].copy_(pd)
        
        mask[i,:N_batch[i],:N_batch[i]] = 1
        mask_lg[i,:E_batch[i],:E_batch[i]] = 1
        
    return X, W, T, XL, WL, Pm, Pd, mask, mask_lg, N_batch, E_batch

"""
def batch_normalization(H, N_batch, mask):
    #""Batch normalization layer taking into account the padded nodes
    
    #nput : H is a tensor (bs, n_features, Nmax)
    
   output : H_norm, avg and var of each feature
    #""

    avg = mean_with_padding(H, N_batch,mask)
    H = H - avg
    
    var = 10**-20 + mean_with_padding(H ** 2, N_batch, mask)
    H = H / var.sqrt()
    
    return H, avg, var
"""

def spatial_normalization(H, N_batch, mask):
    
    avg = spatial_mean_with_padding(H, N_batch,mask)
    H = (H.transpose(2,0)).transpose(2,1)
    #print(H.shape)
    #print(avg.shape)
    H = H - avg
    H = (H.transpose(2,1)).transpose(2,0)
    var = 10**-15 + spatial_mean_with_padding(H ** 2, N_batch, mask)
    #print(var.shape)
    H = (H.transpose(2,0)).transpose(2,1)
    H = H / var.sqrt()
    H = (H.transpose(2,1)).transpose(2,0)
    return H, avg, var


def spatial_mean_with_padding(tensor, N_batch, mask):
    """ Get mean of tensor (bs, n_features, Nmax)
    accounting for zero padding of batches
    
    output (bs, n_features) """
    
    tensor = mask_embedding(tensor, mask)
    somme = torch.sum(tensor, dim=2)
    n = torch.tensor(N_batch).type(dtype)
    #print(somme)
    #print(tensor.shape)
    #print(somme.shape)
    #print(n.shape)
    return (somme.t() / n).t()
    
def mean_with_padding(tensor, N_batch, mask):
    """ Get mean of tensor, accounting for zero padding of batches"""
    
    tensor = mask_embedding(tensor, mask)
    somme = torch.sum(tensor, dim=0)
    #print(somme)
    nz = 10**-10 + torch.sum((tensor != 0),dim=0).type(dtype)
    #print(nz)
    return somme / nz


def mask_embedding(H, mask):
    
    bs = mask.shape[0]
    N = mask.shape[1]
    nb_feat = H.shape[1]
    
    temp = (mask[:,:,0].view(bs,1,N)).repeat(1,nb_feat,1)
    H = torch.mul(H,temp)
    
    return H
    
    








