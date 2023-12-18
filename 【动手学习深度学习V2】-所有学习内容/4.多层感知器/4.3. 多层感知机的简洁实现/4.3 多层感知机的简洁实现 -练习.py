#!/usr/bin/env python
# coding: utf-8

# # 多层感知机的简洁实现

# In[6]:


import torch
from torch import nn
from d2l import torch as d2l


# ## 模型

# In[7]:


net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),  # 新增隐藏层
    nn.ReLU(),            # 新增隐藏层的激活函数
    nn.Linear(128, 10)
)
lr = 0.05  # 修改学习率

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);


# In[8]:


batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=lr)


# In[ ]:


train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)


# In[ ]:




