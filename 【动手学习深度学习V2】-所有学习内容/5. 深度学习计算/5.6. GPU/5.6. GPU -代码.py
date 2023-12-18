#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())


# In[2]:


get_ipython().system('nvidia-smi')


# In[3]:


import torch
from torch import nn

torch.device('cpu'), torch.device('cuda'), torch.device('cuda:1')


# In[4]:


torch.cuda.device_count()


# In[5]:


def try_gpu(i=0):  #@save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():  #@save
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

try_gpu(), try_gpu(10), try_all_gpus()


# In[6]:


x = torch.tensor([1, 2, 3])
x.device


# In[7]:


X = torch.ones(2, 3, device=try_gpu())
X


# In[8]:


Y = torch.rand(2, 3, device=try_gpu(1))
Y


# In[13]:


Z = X.cuda(0)
print(X)
print(Z)


# In[14]:


Y + Z


# In[15]:


Z.cuda(1) is Z


# In[16]:


net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu())


# In[17]:


net(X)


# In[18]:


net[0].weight.data.device


# 

# In[ ]:




