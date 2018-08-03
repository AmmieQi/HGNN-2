#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 16:17:10 2018

@author: sulem
"""

import numpy as np

class Atom():
    
    def __init__(self,index,symb,Z,coord=None,pc=0.0,acceptor=0,donor=0,aromatic=0,
                 hybrid=None,nbH=0,label=None):
    
        self.index = index
        self.symb = symb
        self.Z = Z
        self.coord = coord
        self.partial_charge = pc
        self.acc = acceptor
        self.don = donor
        self.arom = aromatic
        self.hybrid = hybrid
        self.nbH = nbH
        self.label = label
        
class Chemic_bond():
    
    def __init__(self,origin,end,bond=1.0,distance=0.0):
        
        self.origin = origin
        self.end = end
        self.bond = bond
        self.dist = distance
        
