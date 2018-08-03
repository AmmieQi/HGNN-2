#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 16:15:48 2018

@author: sulem
"""

import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F


class GRUUpdate(nn.Module):
    
  def __init__(self, fmap_in, fmap_out):
    super(GRUUpdate, self).__init__()
    self.ih = nn.Linear(fmap_in, 3 * fmap_out)
    self.hh = nn.Linear(fmap_out, 3 * fmap_out)

  def forward(self, i, h):
    r_i, z_i, n_i = self.ih(i).chunk(3,-1)
    r_h, z_h, n_h = self.hh(h).chunk(3,-1)

    r = F.sigmoid(r_i+r_h)
    z = F.sigmoid(z_i+z_h)
    n = F.tanh(n_i+r*n_h)

    o = (1-z)*n + z*h

    return o


class Identity(nn.Module):
    
  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, emb_in, emb_update):
      return emb_update