#!/usr/bin/env python
# coding: utf-8

# # 查阅文档
# 

# ## 查找模块中的所有函数和类

# In[2]:


import torch
print(dir(torch.distributions))


# ## 查找特定函数和类的用法

# In[3]:


help(torch.ones)


# In[4]:


torch.ones(4)

