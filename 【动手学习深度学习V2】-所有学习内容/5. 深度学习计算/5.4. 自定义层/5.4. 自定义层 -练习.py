#!/usr/bin/env python
# coding: utf-8

# # 练习
# 1. 设计一个接受输入并计算张量降维的层，它返回$y_k = \sum_{i, j} W_{ijk} x_i x_j$。
# 2. 设计一个返回输入数据的傅立叶系数前半部分的层。

# 1. 设计一个接受输入并计算张量降维的层，它返回$y_k = \sum_{i, j} W_{ijk} x_i x_j$。

# In[2]:


import torch
from torch import nn

class TensorReductionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(TensorReductionLayer, self).__init__()
        self.weights = nn.Parameter(torch.randn(in_features, in_features, out_features))

    def forward(self, x):
        out = torch.zeros(x.shape[0], self.weights.shape[2])
        for k in range(self.weights.shape[2]):
            for i in range(x.shape[1]):
                for j in range(x.shape[1]):
                    out[:, k] += self.weights[i, j, k] * x[:, i] * x[:, j]
        return out

# 示例使用
input_features = 4
output_features = 1
batch_size = 2

tensor_reduction_layer = TensorReductionLayer(input_features, output_features)
X = torch.rand(batch_size, input_features)
output = tensor_reduction_layer(X)
print(output)


# 2. 设计一个返回输入数据的傅立叶系数前半部分的层。

# In[3]:


class FourierTransformLayer(nn.Module):
    def __init__(self):
        super(FourierTransformLayer, self).__init__()

    def forward(self, x):
        # 计算傅立叶变换
        x_fft = torch.fft.fft(x)
        # 返回前半部分的傅立叶系数
        return x_fft[:, :x_fft.shape[1] // 2]

# 示例使用
batch_size = 2
sequence_length = 4

fourier_layer = FourierTransformLayer()
X = torch.rand(batch_size, sequence_length)
output = fourier_layer(X)
print(output)


# In[ ]:




