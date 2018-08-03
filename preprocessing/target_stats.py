#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 15:52:13 2018

@author: sulem
"""

import pickle
import torch
import logging

import sys
sys.path.insert(0, '/misc/vlgscratch4/BrunaGroup/sulem/chem/HGNN')

import preprocessing.preprocessing as pre

stats_path = '/misc/vlgscratch4/BrunaGroup/sulem/chem/data/tensors/target_stat.pickle'
n_tasks = 13
accuracies = torch.tensor([0.1, 0.05, 0.043, 0.043, 0.043, 0.043,
                         0.043, 0.1, 10.0, 1.2, 0.043, 0.043, 0.0012])

def main():
    
    with open('/misc/vlgscratch4/BrunaGroup/sulem/chem/data/molecules/dataset.pickle','rb') as file :
        molecules = pickle.load(file)

    n = len(molecules)
    targets = torch.zeros(n, 13)
    
    for i in range (n):
        targets[i,:] = pre.molecule_to_instance(molecules[i])[2]

    means = torch.mean(targets,dim=0)
    stds = torch.std(targets,dim=0)
    
    print("Target means and stds : ")
    print(means)
    print(stds)
    
    with open(stats_path,'wb') as file :
        pickle.dump([means,stds,accuracies],file)
        
    logging.info("Stats successfully extracted")
        
        
if __name__ == '__main__':
    main()