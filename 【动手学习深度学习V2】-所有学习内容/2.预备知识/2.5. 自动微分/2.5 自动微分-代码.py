#!/usr/bin/env python
# coding: utf-8

# # 自动微分

# In[1]:


import torch

x = torch.arange(4.0)
x


# In[2]:


x.requires_grad_(True)  # 等价于x=torch.arange(4.0,requires_grad=True)
x.grad  # 默认值是None


# In[3]:


y = 2 * torch.dot(x, x)
y


# In[4]:


y.backward()
x.grad


# In[5]:


x.grad == 4 * x


# In[6]:


# 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
x.grad.zero_()
y = x.sum()
y.backward()
x.grad


# ## 非标量变量的反向传播

# In[7]:


# 对非标量调用backward需要传入一个gradient参数，该参数指定微分函数关于self的梯度。
# 本例只想求偏导数的和，所以传递一个1的梯度是合适的
x.grad.zero_()
y = x * x
# 等价于y.backward(torch.ones(len(x)))
y.sum().backward()
x.grad


# ## 分离计算

# In[8]:


x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
x.grad == u


# In[9]:


x.grad.zero_()
y.sum().backward()
x.grad == 2 * x


# ## Python控制流的梯度计算

# In[10]:


def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c


# In[11]:


a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()


# In[12]:


a.grad == d / a

