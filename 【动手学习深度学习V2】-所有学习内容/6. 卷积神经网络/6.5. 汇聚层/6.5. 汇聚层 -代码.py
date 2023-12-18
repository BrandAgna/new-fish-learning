#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn
from d2l import torch as d2l


# In[2]:


def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y


# In[3]:


X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
pool2d(X, (2, 2))


# In[4]:


pool2d(X, (2, 2), 'avg')


# In[5]:


X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
X


# In[6]:


pool2d = nn.MaxPool2d(3)
pool2d(X)


# In[7]:


pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)


# In[8]:


pool2d = nn.MaxPool2d((2, 3), stride=(2, 3), padding=(0, 1))
pool2d(X)


# In[9]:


X = torch.cat((X, X + 1), 1)
X


# In[10]:


pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)

