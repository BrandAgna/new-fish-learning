#!/usr/bin/env python
# coding: utf-8

# ## 绘制函数$y = f(x) = x^3 - \frac{1}{x}$和其在$x = 1$处切线的图像。

# In[2]:


import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**3 - 1/x

x = np.linspace(-2, 2, 100)  # 创建一个数值范围，避免x=0导致除以0的情况
y = f(x)

# f(x)在x=1处的切线
def tangent_line(x):
    return 4 * x - 4  # f'(x) = 3x^2 + 1/x^2，在x=1处，斜率为4

# 绘制函数图像
plt.plot(x, y, label='y=f(x)')

# 绘制切线
plt.plot(x, tangent_line(x), label='Tangent at x=1', linestyle='--')
plt.legend()
plt.show()


# ## 求函数$f(\mathbf{x}) = 3x_1^2 + 5e^{x_2}$的梯度。

# In[4]:


import torch

x = torch.tensor([1.0, 1.0], requires_grad=True)
f = 3 * x[0] ** 2 + 5 * torch.exp(x[1])
f.backward()
print(x.grad)  # 输出梯度


# ## 函数$f(\mathbf{x}) = \|\mathbf{x}\|_2$的梯度是什么？

# In[5]:


x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
f = torch.norm(x)
f.backward()
print(x.grad)  # 输出梯度


# ## 尝试写出函数$u = f(x, y, z)$，其中$x = x(a, b)$，$y = y(a, b)$，$z = z(a, b)$的链式法则。

# In[7]:


a, b = torch.tensor(1.0, requires_grad=True), torch.tensor(2.0, requires_grad=True)
x = a ** 2 + b
y = a * b + b ** 2
z = a + b

u = x + y + z
u.backward()

print(a.grad)  # du/da
print(b.grad)  # du/db

