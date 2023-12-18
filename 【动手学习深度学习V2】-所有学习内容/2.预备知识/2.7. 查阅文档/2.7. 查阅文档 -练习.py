#!/usr/bin/env python
# coding: utf-8

# ## 练习
# 
# 1. 在深度学习框架中查找任何函数或类的文档。请尝试在这个框架的官方网站上找到文档。

# In[4]:


import torch

# 例如，获取torch.nn.Module类的文档
help(torch.nn.Module)


# In[5]:


# 在Jupyter Notebook中查看文档
get_ipython().run_line_magic('pinfo', 'torch.nn.Module')


# In[6]:


import webbrowser

# 打开torch.nn.Module的在线文档页面
webbrowser.open('https://pytorch.org/docs/stable/nn.html#module')


# In[ ]:




