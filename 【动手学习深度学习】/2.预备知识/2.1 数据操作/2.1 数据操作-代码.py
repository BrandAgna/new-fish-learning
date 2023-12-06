#!/usr/bin/env python
# coding: utf-8

# In[10]:


import torch


# ## 入门

# In[2]:


x = torch.arange(12)
x


# In[3]:


x.shape


# In[4]:


x.numel()


# In[5]:


X = x.reshape(3, 4)
X


# In[6]:


torch.zeros((2, 3, 4))


# In[7]:


torch.ones((2, 3, 4))


# In[8]:


torch.randn(3, 4)


# In[9]:


torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])


# ## 运算符

# In[11]:


x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # **运算符是求幂运算


# In[12]:


torch.exp(x)


# In[13]:


X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)


# In[14]:


X == Y


# In[15]:


X.sum()


# ## 广播机制

# In[17]:


a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
a, b


# In[18]:


a + b


# ## 索引和切片

# In[19]:


X[-1], X[1:3]


# In[20]:


X[1, 2] = 9
X


# In[21]:


X[0:2, :] = 12
X


# ## 节省内存

# In[22]:


before = id(Y)
Y = Y + X
id(Y) == before


# In[23]:


Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))


# In[24]:


before = id(X)
X += Y
id(X) == before


# ## 转换为其他Python对象

# In[25]:


A = X.numpy()
B = torch.tensor(A)
type(A), type(B)


# In[26]:


a = torch.tensor([3.5])
a, a.item(), float(a), int(a)


# In[ ]:




