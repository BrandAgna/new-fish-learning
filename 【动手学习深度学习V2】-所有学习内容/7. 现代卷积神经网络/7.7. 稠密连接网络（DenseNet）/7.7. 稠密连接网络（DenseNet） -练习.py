#!/usr/bin/env python
# coding: utf-8

# ## 练习
# 1. 为什么我们在过渡层使用平均汇聚层而不是最大汇聚层？
# 1. DenseNet的优点之一是其模型参数比ResNet小。为什么呢？
# 1. DenseNet一个诟病的问题是内存或显存消耗过多。
#     1. 真的是这样吗？可以把输入形状换成$224 \times 224$，来看看实际的显存消耗。
#     1. 有另一种方法来减少显存消耗吗？需要改变框架么？
# 1. 实现DenseNet论文 :cite:`Huang.Liu.Van-Der-Maaten.ea.2017`表1所示的不同DenseNet版本。
# 1. 应用DenseNet的思想设计一个基于多层感知机的模型。将其应用于 :numref:`sec_kaggle_house`中的房价预测任务。 

# ### 练习1. 为什么我们在过渡层使用平均汇聚层而不是最大汇聚层？
# 
# 在DenseNet的过渡层中，使用平均汇聚层（average pooling）而不是最大汇聚层（max pooling），主要是因为平均汇聚在减少空间维度（如高度和宽度）的同时，有助于保留更多的背景信息或特征。最大汇聚层通过选择最显著的特征（通常是最强的信号），可能会丢失一些重要信息，特别是在DenseNet这种紧密连接的架构中。平均汇聚通过考虑所有特征，有助于在网络的这一部分保持更多的信息。
# 
# ### 练习2. DenseNet的优点之一是其模型参数比ResNet小。为什么呢？
# 
# DenseNet的参数量相对较小的原因在于其使用的特征重用机制。在DenseNet中，每个层都直接与之前所有层相连，这意味着网络不需要重新学习之前层已经学到的特征。由于这种密集连接，每个层只需要产生相对较少的特征图，从而减少了参数的数量。此外，DenseNet在其过渡层中通过减半通道数来进一步减少参数和计算量。
# 
# ### 练习3. DenseNet的显存消耗问题
# 
# #### A. 真的是这样吗？可以把输入形状换成$224 \times 224$，来看看实际的显存消耗。
# 
# 由于DenseNet在每个层中将前面所有层的输出合并，因此随着网络深度的增加，所需的显存量会迅速增加。这是因为网络需要存储所有层的输出，以便用于后续层的输入。您可以通过实验，将输入形状更改为$224 \times 224$来观察显存消耗的增加。
# 
# #### B. 有另一种方法来减少显存消耗吗？需要改变框架么？
# 
# 减少显存消耗的一种方法是降低网络的成长率（growth rate），即每个Dense块产生的通道数。此外，可以通过减少每个Dense块中的层数来降低显存需求。还可以通过使用深度可分离卷积来替换普通卷积来减少模型的参数和显存占用。这些更改不需要改变整体架构框架，只需要调整网络的特定部分。

# In[2]:


## 练习4
import torch
from torch import nn

# 定义卷积块
def conv_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1))

# 定义Dense块
class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1)
        return X

# 定义过渡层
def transition_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2))

# 构建DenseNet-121
def densenet121():
    b1 = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    # 根据DenseNet-121的结构，设置每个Dense块中的卷积层数
    num_channels, growth_rate = 64, 32
    num_convs_in_dense_blocks = [6, 12, 24, 16]

    blks = []
    for i, num_convs in enumerate(num_convs_in_dense_blocks):
        blks.append(DenseBlock(num_convs, num_channels, growth_rate))
        num_channels += num_convs * growth_rate
        if i != len(num_convs_in_dense_blocks) - 1:
            blks.append(transition_block(num_channels, num_channels // 2))
            num_channels = num_channels // 2

    net = nn.Sequential(
        b1, *blks,
        nn.BatchNorm2d(num_channels), nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(num_channels, 10))
    
    return net

# 实例化模型
model_densenet121 = densenet121()


# In[3]:


## 练习5
import torch
from torch import nn
import torch.nn.functional as F

# 定义DenseLayer
class DenseLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(DenseLayer, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))

# 定义基于DenseNet思想的MLP模型
class DenseMLP(nn.Module):
    def __init__(self, input_size, growth_rate, num_layers, output_size):
        super(DenseMLP, self).__init__()
        self.layers = nn.ModuleList()
        current_size = input_size

        for i in range(num_layers):
            self.layers.append(DenseLayer(current_size, growth_rate))
            current_size += growth_rate

        self.final_linear = nn.Linear(current_size, output_size)

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_features = layer(torch.cat(features, dim=1))
            features.append(new_features)

        return self.final_linear(torch.cat(features, dim=1))

# 示例：定义一个模型用于房价预测
input_size = 15  # 假设输入特征的数量为15
growth_rate = 10
num_layers = 5
output_size = 1  # 房价预测是一个回归任务

model = DenseMLP(input_size, growth_rate, num_layers, output_size)


# In[ ]:




