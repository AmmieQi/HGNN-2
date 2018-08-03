#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 16:23:37 2018

@author: sulem
"""
from os import listdir
from os.path import isfile, join
import torch
from random import shuffle
import pickle

from preprocessing.preprocessing import graph_operators

save_path = '/misc/vlgscratch4/BrunaGroup/sulem/chem/data/generated'


def load_graph_sets(n=1000, Nmax=50, d=5, p=0.5, c=0.5):
    
    Ntrain = int(0.8 * n)
    Nvalid = int(0.1 * n)
    Ntest = n - Nvalid - Ntrain
    
    train_path = join(save_path, 'cp_train_' + str(Ntrain) + '.pickle')
    test_path =  join(save_path, 'cp_test_' + str(Ntest) + '.pickle')
    valid_path =  join(save_path, 'cp_valid_' + str(Nvalid) + '.pickle')
    data = three_collinear_points(n, Nmax, d, p, c)
    
    # Training set
    train_set = data[:Ntrain]
    # Validation set
    valid_set = data[Ntrain:Ntrain+Nvalid]
    # Test set
    test_set = data[Ntrain+Nvalid:]

    with open(train_path,'wb') as fileout:
        pickle.dump(train_set,fileout)
    with open(test_path,'wb') as fileout:
        pickle.dump(test_set,fileout)
    with open(valid_path,'wb') as fileout:
        pickle.dump(valid_set,fileout)
    

def three_collinear_points(n, Nmax, d, p, c):

    """ Creates a dataset of n random graphs (with random number of nodes)
    that contain 3 collinear points with probability p. The adjacency is random
    with probability c of edge between 2 nodes. d is the dimension of the nodes
    features """
    
    data = []
    y = (torch.rand(n) < p)
    N = torch.randint(low = 0, high = Nmax - 3, size=[n])
    
    for i in range (n):
        
        Ni = int(N[i].item())
        #print(Ni)
        
        if y[i] == 1:
            x = torch.randn(1,d)
            x1 = 10*torch.randn(1) * x
            x2 = 10*torch.randn(1) * x
            x3 = 10*torch.randn(1) * x
            
            Xtemp = torch.cat((torch.randn(Ni, d), x1, x2, x3),dim=0)
            idx = list(range(Ni+3))
            shuffle(idx)
            
            X = torch.zeros(3+Ni, d)
            for j,ind in enumerate(idx):
                X[j,:] = Xtemp[ind,:]

        else :        
            X = torch.randn(3+Ni, d)
            
        A = (torch.rand(3+Ni, 3+Ni) > c).type(torch.FloatTensor)
        A[0,1] = 1.0
        A = torch.min(A + A.t(), torch.ones(3+Ni))
        #print(A.shape)
        
        W, WL, Pm, Pd = graph_operators([X, A],dual=True)
        
        data.append([X,A,torch.Tensor([y[i].item()]).type(torch.LongTensor), W, WL, Pm, Pd])
        
    return data

    