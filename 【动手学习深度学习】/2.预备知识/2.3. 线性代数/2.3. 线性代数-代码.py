#!/usr/bin/env python
# coding: utf-8

# # 线性代数

# ## 标量

# In[1]:


import torch

x = torch.tensor(3.0)
y = torch.tensor(2.0)

x + y, x * y, x / y, x**y


# ## 向量

# In[2]:


x = torch.arange(4)
x


# In[3]:


x[3]


# ### 长度、维度和形状

# In[4]:


len(x)


# In[5]:


x.shape


# ## 矩阵

# In[6]:


A = torch.arange(20).reshape(5, 4)
A


# In[7]:


A.T


# In[8]:


B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
B


# In[9]:


B == B.T


# ## 张量

# In[10]:


X = torch.arange(24).reshape(2, 3, 4)
X


# ## 张量算法的基本性质

# In[11]:


A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()  # 通过分配新内存，将A的一个副本分配给B
A, A + B


# In[12]:


A * B


# In[13]:


a = 2
X = torch.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape


# ## 降维

# In[17]:


x = torch.arange(4, dtype=torch.float32)
x, x.sum()


# In[18]:


A.shape, A.sum()


# In[19]:


A_sum_axis0 = A.sum(axis=0)
A_sum_axis0, A_sum_axis0.shape


# In[20]:


A_sum_axis1 = A.sum(axis=1)
A_sum_axis1, A_sum_axis1.shape


# In[21]:


A.sum(axis=[0, 1])  # 结果和A.sum()相同


# In[22]:


A.mean(), A.sum() / A.numel()


# In[23]:


A.mean(axis=0), A.sum(axis=0) / A.shape[0]


# ### 非降维求和

# In[24]:


sum_A = A.sum(axis=1, keepdims=True)
sum_A


# In[25]:


A / sum_A


# In[26]:


A.cumsum(axis=0)


# ## 点积（Dot Product）

# In[27]:


y = torch.ones(4, dtype = torch.float32)
x, y, torch.dot(x, y)


# In[28]:


torch.sum(x * y)


# ## 矩阵-向量积

# In[29]:


A.shape, x.shape, torch.mv(A, x)


# ## 矩阵-矩阵乘法

# In[30]:


B = torch.ones(4, 3)
torch.mm(A, B)


# ## 范数

# In[31]:


u = torch.tensor([3.0, -4.0])
torch.norm(u)


# In[32]:


torch.abs(u).sum()


# In[33]:


torch.norm(torch.ones((4, 9)))

