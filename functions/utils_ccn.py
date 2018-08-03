#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 15:06:03 2018

@author: sulem
"""

import pdb
import numpy as np
import torch
#import functions.contract18 # for cuda contraction
import torch.nn.functional as Func
from torch.autograd import Variable
import torch.nn as nn
from functions.contraction import collapse6to3

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dtype_l = torch.cuda.LongTensor
    #torch.cuda.manual_seed(0)
else:
    dtype = torch.FloatTensor
    dtype_l = torch.LongTensor
    #torch.manual_seed(0)


class CompnetUtils():
    
    def __init__(self, cudaflag=False):
        '''
        Wrapper class that contains useful methods for computing various the
        base feature, feature updates for input graphs
        '''
        self.cudaflag = cudaflag

        def python_contract(T, adj):
            '''
            T is a Variable containing a 4-d tensor of size (n, n, n, channels)
            adj: Variable containing a tensor of size (n, n)
            '''
            T = T.permute(3, 0, 1, 2)
            H = self.tensorprod(T, adj)
            
            return collapse6to3(H)
        
        """

        if cudaflag:
            self.outer_contract = functions.contract18.Contract18Module().cuda()
        else:
            self.outer_contract = python_contract
        """
        self.outer_contract = python_contract
        

    def tensorprod(self, T, A):
        d1 = len(T.data.shape)
        d2 = len(A.data.shape)
        for i in range(d2):
            T = torch.unsqueeze(T, d1+i)
        
        return T*A


    def _get_chi(self, i, j):
        '''
        i: int representing a vertex
        j: int representing a vertex
        Computes the chi tensor j -> i of size di x dj:
            chi[1,k] = 1 if k = j
        '''
        
        di = self.deg[i].item()
        dj = self.deg[j].item()
        chi = torch.zeros(di,dj).type(dtype)
        
        for k in range (di):
            ind_i = self.neighbors[i][k].item()
            ind_j = ((self.neighbors[j]==ind_i).nonzero())
            if not ind_j.shape == torch.Size([0]): # index of the neighbor k of i amongst the neighbors of j
                chi[k,ind_j.item()] = 1
        
        #print(i,j,di,dj,chi.shape)
        
        """
        def _slice_matrix(i, j):
            '''
            Helper function to compute the chi
            '''
            n = self.adj.shape[0]
            il = [ii for ii in range(n) if self.adj[i, ii] > 0] # i _ratioand neighbors of i
            jl = [jj for jj in range(n) if self.adj[j, jj] > 0] # j and neighbors of j
            chi = np.identity(n)[il, :] # rows corresponding to neighbors of i
            # columns corresponding to neighbors of j. will be 1 if theyre the same, 0 else
            
            return chi[:, jl]

        ret = Variable(torch.from_numpy(_slice_matrix(i, j)).float(), requires_grad=False)
        
        #print("Chi ({},{}) dimensions : {}".format(i,j,ret.shape))
        #print("Nb of non zero coeff : {}".format(torch.nonzero(ret).shape[0]))
        return ret.cuda() if self.cudaflag else ret
        """
        
        return chi
    
    
    def _get_chi_root(self, i):
        '''
        Get the chi matrix between the full graph and vertex i's receptive field.
        i: int
        Returns Variable of a tensor of size n x deg(i), where n = size of the graph
        '''
        n = self.A.shape[0]
        di = self.deg[i].item()
        chi_root = torch.zeros(n,di).type(dtype)
        for k in range(di):
            j = self.neighbors[i][k].item()
            chi_root[j,k] = 1
            
        return chi_root.cuda() if self.cudaflag else chi_root

    
    def _register_chis(self, A):
        '''
        Store the chi matrices for each pair of vertices for later use.
        adj: numpy adjacency matrix
        Returns: list of list of Variables of torch tensors
        The (i, j) index of this list of lists will be the chi matrix for vertex i and j
        '''
        n = A.shape[0]
        self.chis = []
        for i in range(n):
            chi = []
            for j in range(n):
                if A[i][j] > 0 :
                    chi.append(self._get_chi(i, j))
                else :
                    chi.append(None)
            
            chi.append(self._get_chi_root(i))
            self.chis.append(chi)
        #print("Chi matrices dimensions : {}, {}".format(len(self.chis),len(self.chis[0])))
        return self.chis

    
    def get_F0(self, X, A):
        '''
        Computes the base features for CCN 2D
        X: numpy matrix of size n x input_feats
        adj: numpy array of size n x n
        Returns a list of Variables(tensors)
        '''

        self.A = A
        n = A.shape[0]
        d = torch.sum((A > 0.0),dim=1) # vector of nodes' degrees
        N = [] # list of neighbours (= receptive field) for each node
        for i in range (n):
            Ni = torch.nonzero(A[i,:]).type(dtype_l)
            N.append(torch.Tensor([Ni[j,0] for j in range (d[i].item())]).type(dtype_l))
        self.neighbors = N
        self.deg = d
        self._register_chis(A)
        
        F_0 = []
        for i in range(n):
            di = d[i].item()
            f = torch.cat([X[i,:].view(1,-1) for j in range(di)], dim=0)
            f = torch.cat([f.view(1,di,-1) for j in range(di)], dim=0)
            F_0.append(f)

        # In the paper, the base features should be of size 1x1xchannels, but
        # we initialize it to n_j x n_j x channels(where n_j is the size of
        # the 1-hop neighborhood of vertex j) for simplicity to avoid
        # two cases for the promotion step.

        if self.cudaflag:
            F_0 = [f.cuda() for f in F_0]

        return F_0

    
    def get_F0_1D(self, X, A):
        '''
        Computes the base features for CCN 1D
        X: tensor of size n x input_feats
        adj: tensor of size n x n
        Returns a list of Variables(tensors)
        '''
        
        self.A = A
        n = A.shape[0]
        d = torch.sum((A > 0.0),dim=1) # vector of nodes' degrees
        N = [] # list of neighbours (= receptive field) for each node
        for i in range (n):
            Ni = torch.nonzero(A[i,:]).type(dtype_l)
            N.append(torch.Tensor([Ni[j,0] for j in range (d[i].item())]).type(dtype_l))
        self.neighbors = N
        self.deg = d
        self._register_chis(A)
 
        """
        F_0 = []
        for i in range(n):
            f = X[i,:]
            f = f.view(1,-1)
            F_0.append(f)
        """
        
        F_0 = []
        for i in range(n):
            f = torch.cat([X[i,:].view(1,-1) for j in range(d[i])],dim=0)
            #print(f.shape)
            F_0.append(f)
        

        if self.cudaflag:
            F_0 = [f.cuda() for f in F_0]

        return F_0

    
    def _promote(self, F_prev, i, j):
        '''
        Promotes the previous level's feature vector of vertex j by doing: chi * F * chi.T
        F_prev: a list of 3-D tensors of size (rows, cols, channels)
        Returns a Variable containing a tensor of size nbrs(i) x nbrs(i) x channels
        '''

        # In the paper, the receptive field of vertex i would be the union of receptive fields
        # of its neighbors. But here we just promote according to the vertices
        # in i's 1-hop neighborhood for simplicity(and to avoid computing this union of
        # receptive fields.
        
        ret = torch.matmul(self.chis[i][j], torch.matmul(F_prev[j].permute(2, 0, 1), self.chis[i][j].t()))
        # move channel index back to the back
        return ret.permute(1, 2, 0)

    
    def _promote_1D(self, F_prev, i, j):
        '''
        Promotion for 1D CCN.
        '''
        #print("Promote {},{}".format(i,j))
        #print(F_prev[j].shape)
        #print(F_prev[i].shape)
        #print(self.chis[i][j].shape)
        ret = torch.matmul(self.chis[i][j], F_prev[j])
        
        return ret

    
    def get_nbr_promotions(self, F_prev, i):
        '''
        Promotes the neighbors of vertex i and stacks them into a tensor for CCN 2D
        F_prev: list of tensors
        i: int(representing a vertex)
        Returns a Variable containing a tensor of size nbrs(i) x nbrs(i) x nbrs(i) x channels
        '''
        n = self.A.shape[0]
        # Promotions of neighbors of vertex i
        all_promotions = [self._promote(F_prev, i, j) for j in range(n) if self.A[i, j] > 0]
        T = torch.stack(all_promotions, 0)
        return T

    
    def get_nbr_promotions_1D(self, F_prev, i):
        """
        Returns a tensor T of size di x di x d
        """
        
        n = self.A.shape[0]
        promotions = [self._promote_1D(F_prev, i, j) for j in range(n) if self.A[i, j] > 0]
        T = torch.stack(promotions, 0)
        
        return T

    
    def update_F(self, F_prev, W):
        '''
        Vertex feature update for CCN 2D. This performs the feature update for
        all vertices.
        F_prev: list of Variables containing a tensor of each vertex' state
        W: linear layer
        Returns a list of Variables of tensors of each vertex' new state
        '''
        assert len(F_prev) == self.A.shape[0]

        def single_vtx_update(F_prev, i, W):
            T_i = self.get_nbr_promotions(F_prev, i)
            collapsed = self.outer_contract(T_i, self.chis[i][i])
            ret = W(collapsed)
            
            return Func.relu(ret)

        F_new = [single_vtx_update(F_prev, i, W) for i in range(len(F_prev))]
        
        return F_new

    
    def update_F_1D(self, F_prev, W):
        '''
        Vertex feature update for 1D CCN. This performs the feature update for
        all vertices.
        '''

        def single_vtx_update(F_prev, i, W):
            #print("Node {}".format(i))
            T_i = self.get_nbr_promotions_1D(F_prev, i)
            #print("Promotion T : dim {} values {}".format(T_i.shape,T_i))
            row_contract = T_i.sum(0)
            col_contract = T_i.sum(1)
            collapsed = torch.cat([row_contract, col_contract], 1)
            #print(collapsed.shape)
            #print("collapsed : dim {} values {}".format(collapsed.shape,collapsed))
            ret = W(collapsed)
            #print(ret.shape)
            return Func.relu(ret)

        F_new = [single_vtx_update(F_prev, i, W) for i in range(len(F_prev))]
        
        return F_new
    
    
    
    