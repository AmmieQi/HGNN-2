#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 18:07:39 2018

@author: sulem
"""

import torch

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
        #print(AL.shape)
        #print(dl.shape)
        lg_operators[:,:,1] = torch.diag(dl)
        lg_operators[:,:,2].copy_(AL)
        CL = AL.clone()
        for j in range (1,J):
            CL = torch.matmul(CL, CL)
            lg_operators[:,:,j+2].copy_(CL)
            
        return operators, lg_operators, Pm, Pd