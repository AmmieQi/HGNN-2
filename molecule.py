#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 11:20:29 2018

@author: sulem
"""

from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig

from atom import *

class Molecule():
    
    #Constructor
    def __init__(self,atoms=None,bonds=None,tag=None,ident=None,
                 A=0.0,B=0.0,C=0.0,mu=0.0,alpha=0.0,
                 homo=0.0,lumo=0.0,r2=0.0,zpve=0.0,
                 U0=0.0,U=0.0,H=0.0,G=0.0,Cv=0.0,frequencies=None,
                 smile=None,chemMol=None):
        
        self.tag = tag
        self.ident = ident
        
        self.atoms = atoms
        self.bonds = bonds
        if (atoms == None):
            self.Na = 0
        else:   
            self.Na = len(atoms)
        
        self.A = A
        self.B = B
        self.C = C
        self.mu = mu
        self.alpha = alpha
        self.homo = homo
        self.lumo = lumo
        self.gap = lumo - homo
        self.r2 = r2
        self.zpve = zpve
        self.U0 = U0
        self.U = U
        self.H = H
        self.G = G
        self.Cv = Cv
        self.freq = frequencies
        self.smile = smile
        self.chemMol = chemMol
        
    
    def add_atom(self,atom):
        
        if (self.atoms == None):
            self.atoms = []
        self.atoms.append(atom)
        self.Na += 1
        
    
    def add_bond(self,atom1,atom2,bond=1.0,distance=0.0):
        
        self.bonds.append(Chemic_bond(atom1,atom2,bond,distance))
        
    
    def add_atom_coord(self,i,coord):
        self.atoms[i].coord = coord
        
        
    def add_atom_pc(self,i,pc):
        self.atoms[i].pc = pc
        
    
    def get_atomlist(self):
        
        return ([a.symb for a in self.atoms])
        
        
        
        
        
        
        