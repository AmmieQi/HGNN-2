#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 17:15:55 2018

@author: sulem
"""

import torch
import numpy as np
from random import shuffle

#import utils
from functions.operators import graph_operators

# Check if CUDA is enabled
if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dtype_l = torch.cuda.LongTensor
    
else:
    dtype = torch.FloatTensor
    dtype_l = torch.LongTensor


def _divide_batch(nb_samples_in, batch_size, idx):
    """Returns a list of lists of indices of each mini batch given idx"""
    
    if nb_samples_in % batch_size == 0:
        nb_batches = nb_samples_in // batch_size 
    else:
        nb_batches = (nb_samples_in // batch_size) + 1
    idx_list = []
    for i in range(0,nb_batches):
        if i==nb_batches-1 :
            idx_list.append(idx[i*batch_size:])
        else:
            idx_list.append(idx[i*batch_size:(i+1)*batch_size])
    
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
    T = torch.zeros(bs,1)
    XL = torch.zeros(bs , 1, Emax)
    WL = torch.zeros(bs , Emax , Emax , J+2)
    Pm = torch.zeros(bs , Nmax, Emax)
    Pd = torch.zeros(bs , Nmax, Emax)   
    
    # Pad samples with zeros
    for i in range(bs):
        
        #print("Molecule " + str(i))
        x, A, t, w, wl, pm, pd = batch[i]
        
        #print("dual size : " + str(E_batch[i]))
        #print(npad_sizes[i])
        if npad_sizes[i] > 0 :
        
            z1 = torch.zeros(npad_sizes[i].item(), n_features)
            z2 = torch.zeros(N_batch[i].item(), npad_sizes[i].item())
            z3 = torch.zeros(npad_sizes[i].item(), Nmax)
            
            b1 = torch.zeros(N_batch[i].item(), npad_sizes[i].item(), J+2)
            b2 = torch.zeros(npad_sizes[i].item(), Nmax, J+2)
            b3 = torch.zeros(npad_sizes[i].item(), E_batch[i].item())
            
            x = torch.cat((x, z1),dim=0)
            A = torch.cat(((torch.cat((A, z2),dim=1)),z3),dim=0)
            
            w = torch.cat((torch.cat((w, b1),dim=1),b2),dim=0)
            pm = torch.cat((pm, b3),dim=0)
            pd = torch.cat((pd, b3),dim=0)

        #w, wl, pm, pd = graph_operators([x,A],J,True)
        
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
        T[i,0] = t[task]
        XL[i,:,:].copy_(xl)
        W[i,:,:,:].copy_(w)
        WL[i,:,:,:].copy_(wl)
        Pm[i,:,:].copy_(pm)
        Pd[i,:,:].copy_(pd)
        
        mask[i,:N_batch[i],:N_batch[i]] = 1
        mask_lg[i,:E_batch[i],:E_batch[i]] = 1
        
    return X, W, T, XL, WL, Pm, Pd, mask, mask_lg, N_batch, E_batch



    








