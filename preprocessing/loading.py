#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 10:50:46 2018

@author: sulem
"""

from os import listdir
from os.path import isfile, join
from random import shuffle

import numpy as np
import pickle

import preprocessing.preprocessing as pre


def prepare_experiment_sets(data,shuf=False):
    
    n = len(data)
    print(" Size of the data: " + str(n))
        
    Ntrain = int(0.8 * n)
    Nvalid = int(0.1 * n)
    Ntest = n - Nvalid - Ntrain
    
    print("Size of training, validation and test sets : {} {} {}".format(Ntrain, Nvalid, Ntest))
    
    if shuf==True:
        shuffle(data)

    train_set = data[:Ntrain]
    valid_set = data[Ntrain:Ntrain+Nvalid]
    test_set = data[Ntrain+Nvalid:]
    
    return train_set, valid_set, test_set


def split_data(n=10):
    
    path_in = '/misc/vlgscratch4/BrunaGroup/sulem/chem/data/processed/qm9.pickle'
    with open(path_in,'rb') as file:
        data = pickle.load(file)
        
    L = len(data)
    print(L)       
    idx = np.random.permutation(L)
    
    path_out = '/misc/vlgscratch4/BrunaGroup/sulem/chem/data/processed/'
    N = int(len(data) / n)
    
    for k in range (n):

        name = 'qm9_' + str(k) + '.pickle'
        p = join(path_out, name)
        s = []
        if k < n-1:
            limite = (k+1)*N
        else:
            limite = len(data)
            
        for j in range (k*N, limite):
            s.append(data[idx[j]])
        
        with open(p,'wb') as file:
            pickle.dump(s, file)
        
        print('Dataset ' + str(k) + ' successfully saved')
        

def load_qm9(spatial=False,charge=False):
    """Loads all molecules from QM9 and makes it a list of Molecule objects"""
    
    dir_path = '/misc/vlgscratch4/BrunaGroup/sulem/chem/data/dsgdb9nsd.xyz'
    files = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
        
    instances = []
    for f in range(len(files)):
        file = join(dir_path, files[f])
        molecule = pre.xyz_to_molecule(file)
        data = pre.molecule_to_instance(molecule,spatial,charge)
        instances.append(data)
    
    if spatial==True and charge==False:
        path_out = '/misc/vlgscratch4/BrunaGroup/sulem/chem/data/processed/qm9_sp.pickle'
    
    elif spatial==True and charge==True:
        path_out = '/misc/vlgscratch4/BrunaGroup/sulem/chem/data/processed/qm9_sp_ch.pickle'
        
    elif spatial==False and charge==True:
        path_out = '/misc/vlgscratch4/BrunaGroup/sulem/chem/data/processed/qm9_ch.pickle'
        
    else:
        path_out = '/misc/vlgscratch4/BrunaGroup/sulem/chem/data/processed/qm9_new.pickle'
    
    with open(path_out,'wb') as fileout:
        pickle.dump(instances,fileout)
        
        
def load_experiment_sets(Nvalid=0,Ntest=0,Ntrain=0):
    
    with open('/misc/vlgscratch4/BrunaGroup/sulem/chem/data/processed/qm9.pickle','rb') as file :
        data = pickle.load(file)
        
    n = len(data)
    print(n)       
    # Randomize and split into train, validation and test sets
    idx = np.random.permutation(len(data))
    
    if Ntrain > 0 or Ntest > 0 or Nvalid > 0 :
        assert Ntrain + Ntest + Nvalid <= n, "Not enough data available"
        train_path = '/misc/vlgscratch4/BrunaGroup/sulem/chem/data/processed/train_' + str(Ntrain) + '.pickle'
        test_path = '/misc/vlgscratch4/BrunaGroup/sulem/chem/data/processed/test_' + str(Ntest) + '.pickle'
        valid_path = '/misc/vlgscratch4/BrunaGroup/sulem/chem/data/processed/valid_' + str(Nvalid) + '.pickle'
    
    else :
        train_path = '/misc/vlgscratch4/BrunaGroup/sulem/chem/data/processed/train.pickle'
        test_path = '/misc/vlgscratch4/BrunaGroup/sulem/chem/data/processed/test.pickle'
        valid_path = '/misc/vlgscratch4/BrunaGroup/sulem/chem/data/processed/valid.pickle'
        
        Ntrain = int(0.8 * n)
        Nvalid = int(0.1 * n)
        Ntest = n - Nvalid - Ntrain
    
    # Training set
    train_set = []
    for i in range(Ntrain):
        mol =  data[idx[i]]
        train_set.append(mol)
        
    # Validation set
    valid_set = []
    for i in range(Ntrain, Ntrain+Nvalid):
        mol =  data[idx[i]]
        valid_set.append(mol)
        
    # Test set
    test_set = []
    for i in range(Ntrain+Nvalid,n):
        mol =  data[idx[i]]
        test_set.append(mol)  

    with open(train_path,'wb') as fileout:
        pickle.dump(train_set,fileout)
    with open(test_path,'wb') as fileout:
        pickle.dump(test_set,fileout)
    with open(valid_path,'wb') as fileout:
        pickle.dump(valid_set,fileout)

"""
def load_experiment_sets(Nvalid=10000,Ntest=10000,Ntrain=0):
    
    if Ntrain > 0 :
        train_path = '/misc/vlgscratch4/BrunaGroup/sulem/chem/data/tensors/train_' + str(Ntrain) + '.pickle'
        test_path = '/misc/vlgscratch4/BrunaGroup/sulem/chem/data/tensors/test_' + str(Ntest) + '.pickle'
        valid_path = '/misc/vlgscratch4/BrunaGroup/sulem/chem/data/tensors/valid_' + str(Nvalid) + '.pickle'
    
    else :
        train_path = '/misc/vlgscratch4/BrunaGroup/sulem/chem/data/tensors/train.pickle'
        test_path = '/misc/vlgscratch4/BrunaGroup/sulem/chem/data/tensors/test.pickle'
        valid_path = '/misc/vlgscratch4/BrunaGroup/sulem/chem/data/tensors/valid.pickle'
        
    with open('/misc/vlgscratch4/BrunaGroup/sulem/chem/data/molecules/dataset.pickle','rb') as file :
        molecules = pickle.load(file)
    
    # Randomize and split into train, validation and test sets
    idx = np.random.permutation(len(molecules))
    
    # Validation set
    valid_set = []
    for i in range(Nvalid):
        mol =  molecules[idx[i]]
        valid_set.append(pre.molecule_to_instance(mol))
        
    # Test set
    test_set = []
    for i in range(Nvalid,Nvalid+Ntest):
        mol =  molecules[idx[i]]
        test_set.append(pre.molecule_to_instance(mol))  

    # Training set
    train_set = []
    if (Ntrain > 0):
        limite = min(Nvalid+Ntest+Ntrain,len(molecules))
    else :
        limite = len(molecules)
    for i in range(Nvalid+Ntest,limite):
        mol =  molecules[idx[i]]
        train_set.append(pre.molecule_to_instance(mol))  

    with open(train_path,'wb') as fileout:
        pickle.dump(train_set,fileout)
    with open(test_path,'wb') as fileout:
        pickle.dump(test_set,fileout)
    with open(valid_path,'wb') as fileout:
        pickle.dump(valid_set,fileout)
"""        

def load_N_molecules(N, pathout = '/misc/vlgscratch4/BrunaGroup/sulem/chem/data/molecules/debug.pickle'):
    """Loads N molecules from QM9 for debugging"""
    
    dir_path = '/misc/vlgscratch4/BrunaGroup/sulem/chem/data/dsgdb9nsd.xyz'
    files = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
        
    molecules = []
    for f in range(N):
        file = join(dir_path, files[f])
        molecules.append(pre.xyz_to_molecule(file))
    
    with open(pathout,'wb') as fileout:
        pickle.dump(molecules,fileout)
        
    return molecules


def load_molecule(i):
    """Loads molecule with index i from QM9"""
    
    with open('/misc/vlgscratch4/BrunaGroup/sulem/chem/data/molecules/dataset.pickle','rb') as file :
        molecules = pickle.load(file)
        
    mol = [pre.molecule_to_instance(molecules[i])]
    
    dir_path = '/misc/vlgscratch4/BrunaGroup/sulem/chem/data/tensors'
    file = 'molecule' + str(i) + '.pickle'
    file_path = join(dir_path, file)
    
    with open(file_path,'wb') as fileout:
        pickle.dump(mol,fileout)
        
    return mol


