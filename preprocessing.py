#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 12:20:19 2018

@author: sulem
"""

import numpy as np

import os
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import pickle

from graph_reader import *



def molecule_to_instance(mol):
    """ Build 1 input from 1 object molecule 
    V1 : input = atom type, adjacency = bond_type """
    
    #print("Molecule : "+ str(mol.ident))
    #print("Last atom index is : "+ str(mol.atoms[-1].index))
    #print("Molecule size : "+ str(mol.Na))

    instance = torch.zeros(mol.Na,5)
    adjacency = torch.zeros(mol.Na,mol.Na)
    task = torch.zeros(13)
    
    task[0] = mol.alpha
    task[1] = mol.Cv
    task[2] = mol.G
    task[3] = mol.gap
    task[4] = mol.H
    task[5] = mol.homo
    task[6] = mol.lumo
    task[7] = mol.mu
    task[8] = float(mol.freq[-1])
    task[9] = mol.r2
    task[10] = mol.U
    task[11] = mol.U0
    task[12] = mol.zpve
    
    # One-hot encoding
    for i in range (mol.Na):
        s = mol.atoms[i].symb
        
        if (s == 'H'):
            instance[i,0] = 1
            
        elif (s == 'C'):
            instance[i,1] = 1
            
        elif (s == 'N'):
            instance[i,2] = 1

        elif (s == 'O'):
            instance[i,3] = 1
        
        else :
            instance[i,4] = 1
            
    for l in range (len(mol.bonds)):
        
        i = mol.bonds[l].origin
        j = mol.bonds[l].end
        adjacency[i,j] = mol.bonds[l].bond
        adjacency[j,i] = mol.bonds[l].bond
        
    return instance, adjacency, task


def graph_operators(graph,J=1,dual=False):
    """Builds operators matrices for a graph G = (V,A) :
       I,D,A,..,A^(2^J-1),DL,AL,...AL^J"""
        
    V,A = graph
    N = V.shape[0]
    
    # Graph operators on the original graph
    operators = torch.zeros(N,N,J+2)
    operators[:,:,0] = torch.eye(N)

    d = torch.sum(A,dim=1)
    operators[:,:,1] = torch.diag(d.squeeze())
    
    operators[:,:,2].copy_(A)
    C = A.clone()
    for j in range (1,J):
        C = torch.matmul(C, C)
        operators[:,:,j+2].copy_(C)
    
    if (dual == False) :
        return operators
    
    else:
        # Line Graph operators
        M = (A.nonzero()).shape[0]
        #print("Dual size : " + str(M))
        
        lg_operators = torch.zeros(M,M,J+2)
        lg_operators[:,:,0] = torch.eye(M)
        
        AL = torch.zeros(M,M)
        Pm = torch.zeros(N,M)
        Pd = torch.zeros(N,M)
        
        edges = torch.zeros(M,3)
        
        e = 0
        for i in range (N):
            for j in range (i+1,N):
                if (A[i,j] != 0):
                    Pm[i,e]=1
                    Pm[j,e]=1
                    Pd[i,e]=1
                    Pd[j,e]=-1
                    edges[e,0] = i
                    edges[e,1] = j
                    edges[e,2] = A[i,j]
                    e = e+1
                    Pm[i,e]=1
                    Pm[j,e]=1
                    Pd[i,e]=-1
                    Pd[j,e]=1
                    edges[e,0] = j
                    edges[e,1] = i
                    edges[e,2] = A[i,j]
                    
        for m1 in range(M):
            for m2 in range(M):
                if (edges[m1,1] == edges[m2,0] and edges[m1,0] != edges[m2,1]):
                    AL[m1,m2] = edges[m2,2]
                    
        dl = torch.sum(AL,dim=1)
        lg_operators[:,:,1] = torch.diag(dl.squeeze())
        lg_operators[:,:,2].copy_(AL)
        CL = AL.clone()
        for j in range (1,J):
            CL = torch.matmul(CL, CL)
            lg_operators[:,:,j+2].copy_(CL)
            
        return operators, lg_operators, Pm, Pd
                    
    

def load_debug_set():
    
    path = 'data/tensors/debug.pickle'
    
    with open('data/molecules/debug_100.pickle','rb') as file :
        molecules = pickle.load(file)
        
    debug_set = []
    for i in range(len(molecules)):
        mol =  molecules[i]
        debug_set.append(molecule_to_instance(mol))
        
    with open(path,'wb') as fileout:
        pickle.dump(debug_set,fileout)
         

def load_train_valid_sets(Ntrain, Nvalid):
    """Loads small training and validation sets for tuning hyperparameters"""
    
    dir_path = 'data/dsgdb9nsd.xyz'
    files = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
    
    idx = np.random.permutation(len(files))
    
    train_set = []
    for f in range(Ntrain):
        file = join(dir_path, files[idx[f]])
        mol = xyz_to_molecule(file)
        train_set.append(molecule_to_instance(mol))
    
    valid_set = []
    for f in range(Ntrain,Ntrain+Nvalid):
        file = join(dir_path, files[idx[f]])
        mol = xyz_to_molecule(file)
        valid_set.append(molecule_to_instance(mol))
    
    path_train = 'data/tensors/train_set_' + str(Ntrain) + '.pickle'  
    with open(path_train,'wb') as fileout:
        pickle.dump(train_set,fileout)
    
    path_valid = 'data/tensors/valid_set_' + str(Nvalid) + '.pickle'  
    with open(path_valid,'wb') as fileout:
        pickle.dump(valid_set,fileout)
        
    return train_set, valid_set
        
    
def load_experiment_sets(Nvalid=10000,Ntest=10000,Ntrain=0):
    
    train_path = 'data/tensors/train.pickle'
    test_path = 'data/tensors/test.pickle'
    valid_path = 'data/tensors/test.pickle'
    
    with open('data/molecules/dataset.pickle','rb') as file :
        molecules = pickle.load(file)
    
    # Randomize and split into train, validation and test sets
    idx = np.random.permutation(len(molecules))
    
    # Validation set
    valid_set = []
    for i in range(Nvalid):
        mol =  molecules[idx[i]]
        valid_set.append(molecule_to_instance(mol))
        
    # Test set
    test_set = []
    for i in range(Nvalid,Nvalid+Ntest):
        mol =  molecules[idx[i]]
        test_set.append(molecule_to_instance(mol))  

    # Training set
    train_set = []
    if (Ntrain > 0):
        limite = min(Nvalid+Ntest+Ntrain,len(molecules))
    else :
        limite = len(molecules)
    for i in range(Nvalid+Ntest,limite):
        mol =  molecules[idx[i]]
        train_set.append(molecule_to_instance(mol))  

    with open(train_path,'wb') as fileout:
        pickle.dump(train_set,fileout)
    with open(test_path,'wb') as fileout:
        pickle.dump(test_set,fileout)
    with open(valid_path,'wb') as fileout:
        pickle.dump(valid_set,fileout)
        
    return train_set, valid_set, test_set
    
    
    