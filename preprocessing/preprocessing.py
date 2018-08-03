#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 12:20:19 2018

@author: sulem
"""

import numpy as np

import os
import torch

from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig

import sys
sys.path.insert(0, '/misc/vlgscratch4/BrunaGroup/sulem/chem/HGNN')
from preprocessing.molecule import Molecule
from preprocessing.atom import Atom



def molecule_to_instance(mol,spatial=False,charge=False):
    """ Build 1 input from 1 object molecule 
    V1 : input = atom type, adjacency = bond_type """
    
    #print("Molecule : "+ str(mol.ident))
    #print("Last atom index is : "+ str(mol.atoms[-1].index))
    #print("Molecule size : "+ str(mol.Na))
    
    if spatial==True:
        if charge==True:
            instance = torch.zeros(mol.Na,9)
        else:
            instance = torch.zeros(mol.Na,8)
    elif charge==True:
        instance = torch.zeros(mol.Na,6)
    else:
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
            
    if spatial==True:
        instance[i,5] = mol.atoms[i].coord[0]
        instance[i,6] = mol.atoms[i].coord[1]
        instance[i,7] = mol.atoms[i].coord[2]
        if charge==True:
            instance[i,8] = mol.atoms[i].partial_charge
    elif charge==True:
        instance[i,5] = mol.atoms[i].partial_charge

    for l in range (len(mol.bonds)):
        
        i = mol.bonds[l].origin
        j = mol.bonds[l].end
        adjacency[i,j] = mol.bonds[l].bond
        adjacency[j,i] = mol.bonds[l].bond
        
    W, WL, Pm, Pd = graph_operators([instance, adjacency],dual=True)
        
    return instance, adjacency, task, W, WL, Pm, Pd


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
    
    
    