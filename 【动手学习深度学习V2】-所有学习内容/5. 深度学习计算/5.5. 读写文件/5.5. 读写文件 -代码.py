#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn
from torch.nn import functional as F

x = torch.arange(4)
torch.save(x, 'x-file')


# In[2]:


x2 = torch.load('x-file')
x2


# In[3]:


y = torch.zeros(4)
torch.save([x, y],'x-files')
x2, y2 = torch.load('x-files')
(x2, y2)


# In[4]:


mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
mydict2


# In[5]:


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)


# In[6]:


torch.save(net.state_dict(), 'mlp.params')


# In[7]:


clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()


# In[8]:


Y_clone = clone(X)
Y_clone == Y

