#!/usr/bin/env python
# coding: utf-8

# # 概率

# ## 基本概率论

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import torch
from torch.distributions import multinomial
from d2l import torch as d2l


# In[2]:


fair_probs = torch.ones([6]) / 6
multinomial.Multinomial(1, fair_probs).sample()


# In[3]:


multinomial.Multinomial(10, fair_probs).sample()


# In[4]:


# 将结果存储为32位浮点数以进行除法
counts = multinomial.Multinomial(1000, fair_probs).sample()
counts / 1000  # 相对频率作为估计值


# In[5]:


counts = multinomial.Multinomial(10, fair_probs).sample((500,))
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)

d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].numpy(),
                 label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend();

