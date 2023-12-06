#!/usr/bin/env python
# coding: utf-8

# In[11]:


import torch
import numpy as np
import matplotlib.pyplot as plt


# # 问题1
# ## 为什么计算二阶导数比一阶导数的开销要更大？
# 
# ## 答案
# ### 计算二阶导数比一阶导数的开销更大是因为二阶导数需要在一阶导数的基础上进行额外的微分计算。一阶导数本身就涉及到了函数值的变化率计算，而二阶导数则是变化率的变化率，这需要对一阶导数再次应用微分操作。在计算上，这意味着要执行更多的链式法则应用，乘法和加法运算，特别是在涉及多元函数时，每个一阶导数又可能是多个变量的函数，从而导致计算量呈指数级增长。

# # 问题2
# ## 在运行反向传播函数之后，立即再次运行它，看看会发生什么。

# In[12]:


x = torch.arange(4.0, requires_grad=True)
y = 2 * torch.dot(x, x)
y.backward()
try:
    y.backward()  # 尝试再次进行反向传播
except RuntimeError as e:
    print(e)  # 打印错误信息


# ### 因为第一次调用.backward()方法时，PyTorch会自动计算所有梯度并将这些梯度存储在各个张量的.grad属性中。为了节省内存，计算图中的中间变量和缓存被释放和删除。如果没有指定retain_graph=True，那么这个图就会被清空，因此无法再次对其调用.backward()。

# # 问题3
# ## 在控制流的例子中，我们计算d关于a的导数，如果将变量a更改为随机向量或矩阵，会发生什么？
# 
# ## 答案
# ### 当变量a更改为随机向量或矩阵时，计算d关于a的导数会变得更复杂，因为这会涉及到矩阵对矩阵或向量的微分。在这种情况下，我们需要计算雅可比矩阵，而不是单一的导数值。如果控制流依赖于a的值，那么不同元素的变化可能会导致不同的控制流路径，从而影响最终的梯度。

# # 练习问题4
# ## 重新设计一个求控制流梯度的例子，运行并分析结果。

# In[13]:


a = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)
c = torch.where(a > b, a, b)  # 控制流
c.backward()
print(a.grad, b.grad)  # 分析梯度


# # 练习问题5
# ## 使$f(x)=\sin(x)$，绘制$f(x)$和$\frac{df(x)}{dx}$的图像，其中后者不使用$f'(x)=\cos(x)$。

# In[14]:


x = torch.linspace(-2 * np.pi, 2 * np.pi, 200, requires_grad=True)
y = torch.sin(x)
y.sum().backward()  # 对y求和以计算梯度

plt.plot(x.detach().numpy(), y.detach().numpy(), label='f(x) = sin(x)')
plt.plot(x.detach().numpy(), x.grad.numpy(), label="f'(x)")
plt.legend()
plt.show()

