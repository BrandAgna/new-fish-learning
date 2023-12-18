#!/usr/bin/env python
# coding: utf-8

# ## 练习
# 
# 1. 使用 :numref:`sec_model_construction` 中定义的`FancyMLP`模型，访问各个层的参数。
# 1. 查看初始化模块文档以了解不同的初始化方法。
# 1. 构建包含共享参数层的多层感知机并对其进行训练。在训练过程中，观察模型各层的参数和梯度。
# 1. 为什么共享参数是个好主意？

# In[5]:


## 1. 使用 :numref:`sec_model_construction` 中定义的`FancyMLP`模型，访问各个层的参数。
import torch
from torch import nn

class FancyMLP(nn.Module):
    def __init__(self):
        super(FancyMLP, self).__init__()
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        X = torch.mm(X, self.rand_weight) + 1
        X = self.linear(X)
        return X

fancy_net = FancyMLP()
for name, param in fancy_net.named_parameters():
    print(name, param.shape)


# ## 2. 查看初始化模块文档以了解不同的初始化方法。
# 当查看神经网络参数初始化的相关文档时，会发现存在多种方法来初始化模型的权重和偏置。下面将叙述几种常见的初始化方法：
# 
# 1. **零初始化（Zero Initialization）**:
#    - 使用零来初始化权重和偏置。
#    - `nn.init.zeros_(tensor)`：将给定的 tensor 初始化为全零。
# 
# 2. **均匀分布初始化（Uniform Initialization）**:
#    - 权重和偏置初始化为在给定范围内均匀分布的随机数。
#    - `nn.init.uniform_(tensor, a=0.0, b=1.0)`：使用从 `[a, b)` 范围内的均匀分布生成的值来填充 tensor。
# 
# 3. **正态分布初始化（Normal Initialization）**:
#    - 权重和偏置初始化为服从正态分布的随机数。
#    - `nn.init.normal_(tensor, mean=0.0, std=1.0)`：使用指定均值和标准差的正态分布填充 tensor。
# 
# 4. **Xavier/Glorot 初始化**:
#    - 用于保持输入和输出方差一致，适用于传统的激活函数（如sigmoid，tanh）。
#    - `nn.init.xavier_uniform_(tensor)`：使用均匀分布。
#    - `nn.init.xavier_normal_(tensor)`：使用正态分布。
# 
# 5. **He/Kaiming 初始化**:
#    - 特别适用于ReLU激活函数，有助于解决深层网络中的梯度消失或爆炸问题。
#    - `nn.init.kaiming_uniform_(tensor, mode='fan_in', nonlinearity='relu')`：使用均匀分布。
#    - `nn.init.kaiming_normal_(tensor, mode='fan_in', nonlinearity='relu')`：使用正态分布。
# 
# 6. **常数初始化（Constant Initialization）**:
#    - 使用常数来初始化权重和偏置。
#    - `nn.init.constant_(tensor, val)`：使用给定的值 `val` 填充 tensor。
# 
# 每种初始化方法都有其适用的场景和特定的目的，选择哪种方法通常取决于网络的架构和所使用的激活函数。正确的初始化策略可以显著提升模型的学习效率和性能。

# In[7]:


## 3. 构建包含共享参数层的多层感知机并对其进行训练。在训练过程中，观察模型各层的参数和梯度。
# 定义共享层
shared = nn.Linear(8, 8)

# 构建模型
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))

# 模拟输入数据
X = torch.rand(size=(2, 4))

# 模拟训练过程
def train(net, X):
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

    for epoch in range(10):
        output = net(X)
        loss = criterion(output, torch.rand(2, 1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch}: Loss: {loss.item()}')

train(net, X)

# 检查共享层的参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])


# ## 练习 4: 为什么共享参数是个好主意？
# 共享参数可以减少模型的参数数量，从而降低过拟合的风险和计算成本。此外，在处理具有重复模式的数据时（如文本或图像），共享参数可以帮助模型更好地学习这些模式。
