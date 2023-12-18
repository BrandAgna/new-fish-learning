#!/usr/bin/env python
# coding: utf-8

# In[5]:


# 导入所需的库
import torch
from torch import nn
from torch.nn import functional as F


# # 练习 1 如果将`MySequential`中存储块的方式更改为Python列表，会出现什么样的问题？

# ### 如果将 MySequential 中存储块的方式更改为Python列表，可能导致模型的部分功能失效。因为PyTorch的模型是基于Module类构建的，这个类在内部使用OrderedDict来存储层。这样可以确保层的正确注册和参数的更新。如果改为使用Python列表，可能会导致模型在保存、加载、更新参数等操作时出现问题。

# In[15]:


# 练习 2
# 实现一个块，它接收两个块作为输入，并返回它们的串联输出。
class ParallelBlock(nn.Module):
    def __init__(self, block1, block2):
        super().__init__()
        self.block1 = block1
        self.block2 = block2

    def forward(self, X):
        return torch.cat((self.block1(X), self.block2(X)), dim=1)

# 实例化两个简单的网络块
net1 = nn.Linear(20, 30)
net2 = nn.Linear(20, 30)

# 创建平行块
parallel_net = ParallelBlock(net1, net2)

# 测试输入
X = torch.rand(2, 20)

# 输出
output_parallel = parallel_net(X)
print("平行块输出:\n", output_parallel)


# In[16]:


# 练习 3
# 实现一个函数，生成多个相同的块，并在此基础上构建更大的网络。
def make_multiple_blocks(block, num_blocks):
    return nn.Sequential(*[block for _ in range(num_blocks)])

# 实例化一个简单的网络块
block = nn.Linear(20, 20)

# 创建包含多个相同块的网络
multi_block_net = make_multiple_blocks(block, 3)

# 测试输入
X = torch.rand(2, 20)

# 输出
output_multi_block = multi_block_net(X)
print("多块网络输出:\n", output_multi_block)

