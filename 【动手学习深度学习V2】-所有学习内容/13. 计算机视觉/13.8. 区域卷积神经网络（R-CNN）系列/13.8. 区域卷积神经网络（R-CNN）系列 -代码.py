#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision

X = torch.arange(16.).reshape(1, 1, 4, 4)
X


# In[2]:


rois = torch.Tensor([[0, 0, 0, 20, 20], [0, 0, 10, 30, 30]])


# In[3]:


torchvision.ops.roi_pool(X, rois, output_size=(2, 2), spatial_scale=0.1)

