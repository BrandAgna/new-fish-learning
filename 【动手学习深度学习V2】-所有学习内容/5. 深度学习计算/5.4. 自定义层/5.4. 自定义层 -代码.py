#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn.functional as F
from torch import nn


class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()


# In[2]:


layer = CenteredLayer()
layer(torch.FloatTensor([1, 2, 3, 4, 5]))


# In[3]:


net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())


# In[4]:


Y = net(torch.rand(4, 8))
Y.mean()


# In[5]:


class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)


# In[6]:


linear = MyLinear(5, 3)
linear.weight


# In[7]:


linear(torch.rand(2, 5))


# In[8]:


net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
net(torch.rand(2, 64))
