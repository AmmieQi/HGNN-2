#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 10:50:46 2018

@author: sulem
"""

from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig

import os
from os import listdir
from os.path import isfile, join

import logging
import numpy as np
import pickle

from molecule import *


def load_qm9():
    """Loads all molecules from QM9 and makes it a list of Molecule objects"""
    
    dir_path = 'data/dsgdb9nsd.xyz'
    files = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
        
    molecules = []
    for f in range(len(files)):
        file = join(dir_path, files[f])
        molecules.append(xyz_to_molecule(file))
    
    path_out = 'data/molecules/dataset.pickle'
    with open(path_out,'wb') as fileout:
        pickle.dump(molecules,fileout)
        

def load_N_molecules(N, pathout = 'data/molecules/debug.pickle'):
    """Loads N molecules from QM9 for debugging"""
    
    dir_path = 'data/dsgdb9nsd.xyz'
    files = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
        
    molecules = []
    for f in range(N):
        file = join(dir_path, files[f])
        molecules.append(xyz_to_molecule(file))
    
    with open(pathout,'wb') as fileout:
        pickle.dump(molecules,fileout)
        
    return molecules


def load_molecule(i):
    """Loads molecule with index i from QM9"""
    
    dir_path = 'data/dsgdb9nsd.xyz'
    file = 'dsgdb9nsd_0' + str(i) + '.xyz'
    file_path = join(dir_path, file)
        
    molecule = xyz_to_molecule(file_path)
    
    path_out = 'data/molecules/' + str(i) + '.pickle'
    
    with open(path_out,'wb') as fileout:
        pickle.dump(molecule,fileout)
        
    return molecule

"""
def load_dataset(dataset='qm9',Ntest=10000,Nvalid=10000,Ntrain=0):
    
    if dataset == 'qm9':
        
        dir_path = 'data/dsgdb9nsd.xyz'
        files = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
        
        molecules = []
        for f in range(len(files)):
            file = join(dir_path, files[f])
            molecules.append(xyz_to_molecule(file))
        
        # Randomize and split into train, validation and test sets
        idx = np.random.permutation(len(molecules))

        valid_set = [molecules[i] for i in idx[0:Nvalid]]
        test_set = [molecules[i] for i in idx[Nvalid:Nvalid+Ntest]]
        if (Ntrain > 0):
            train_set = [molecules[i] for i in idx[Nvalid+Ntest:Nvalid+Ntest+Ntrain]]
        else:
            train_set = [molecules[i] for i in idx[Nvalid+Ntest:]]
        
        return train_set, valid_set, test_set
    
    logging.warning('Dataset not found')
    return None
"""

def xyz_to_molecule(file):
    
    with open(file,'r') as f:
        
        #Nb of atoms
        Na = int(f.readline())
        
        #Molecule properties
        properties = f.readline()
        prop = properties.split()
        
        #Atom properties
        atomes = []
        for i in range(Na):
            p = f.readline()            
            p = p.replace('.*^', 'e')
            p = p.replace('*^', 'e')
            p = p.split()
            atomes.append(p)
        
        #Frequencies
        freq = f.readline()
        freq = freq.split()
        
        #SMILE
        smile = f.readline()
        smile = smile.split()[0]
        
    mol = smile_to_graph(smile)
    mol.Na = Na
    mol.freq = [float(freq[i]) for i in range (len(freq))]
    mol.tag = prop[0]
    mol.ident = int(prop[1])
    mol.A = float(prop[2])
    mol.B = float(prop[3]) 
    mol.C = float(prop[4]) 
    mol.mu = float(prop[5])
    mol.alpha = float(prop[6]) 
    mol.homo = float(prop[7])
    mol.lumo = float(prop[8]) 
    mol.gap = float(prop[9])
    mol.r2 = float(prop[10])
    mol.zpve = float(prop[11]) 
    mol.U0 = float(prop[12]) 
    mol.U = float(prop[13])
    mol.H = float(prop[14])
    mol.G = float(prop[15])
    mol.Cv = float(prop[16])
    
    for i in range(Na) :
        coord = np.array(atomes[i][1:4]).astype(np.float)
        mol.add_atom_coord(i,coord)
        
        pc=float(atomes[i][4])
        mol.add_atom_pc(i,pc)
    
    for b in mol.bonds:
        i = b.origin
        j = b.end
        dist = np.linalg.norm(mol.atoms[i].coord-mol.atoms[j].coord)
        b.distance = dist
        
    return mol
        
        
def smile_to_graph(smile):
    
    mol = Chem.MolFromSmiles(smile)
    mol = Chem.AddHs(mol)
    fdef_name = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)
    feats = factory.GetFeaturesForMol(mol)
    
    #Construction of the graph
    graph = Molecule()
    
    graph.atoms = []
    graph.bonds = []
    
    for i in range(0, mol.GetNumAtoms()):
        
        atom = mol.GetAtomWithIdx(i)
        node = Atom(i,atom.GetSymbol(),atom.GetAtomicNum(),
                    aromatic=atom.GetIsAromatic(),hybrid=atom.GetHybridization(),
                    nbH=atom.GetTotalNumHs())

        graph.add_atom(node)
        
    for i in range(0, len(feats)):
        
        if feats[i].GetFamily() == 'Donor':
            node_list = feats[i].GetAtomIds()
            for i in node_list:
                graph.atoms[i].don = 1
        
        elif feats[i].GetFamily() == 'Acceptor':
            node_list = feats[i].GetAtomIds()
            for i in node_list:
                graph.atoms[i].acc = 1
                
    for i in range(0, mol.GetNumAtoms()):
        
        for j in range(0, mol.GetNumAtoms()):
            e_ij = mol.GetBondBetweenAtoms(i, j)
            
            if e_ij is not None:
                bond = e_ij.GetBondTypeAsDouble()
                graph.add_bond(i,j,bond)
                
    return graph
                
                
                

        