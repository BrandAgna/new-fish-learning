#!/usr/bin/env python
# coding: utf-8

# ## 练习
# 
# 1. 尝试将平均汇聚层作为卷积层的特殊情况实现。
# 1. 尝试将最大汇聚层作为卷积层的特殊情况实现。
# 1. 假设汇聚层的输入大小为$c\times h\times w$，则汇聚窗口的形状为$p_h\times p_w$，填充为$(p_h, p_w)$，步幅为$(s_h, s_w)$。这个汇聚层的计算成本是多少？
# 1. 为什么最大汇聚层和平均汇聚层的工作方式不同？
# 1. 我们是否需要最小汇聚层？可以用已知函数替换它吗？
# 1. 除了平均汇聚层和最大汇聚层，是否有其它函数可以考虑（提示：回想一下`softmax`）？为什么它不流行？

# ### 1. 尝试将平均汇聚层作为卷积层的特殊情况实现。

# In[10]:


import torch
from torch import nn
import torch.nn.functional as F

class AvgPool2dAsConv2d(nn.Module):
    def __init__(self, kernel_size):
        super(AvgPool2dAsConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.avg_pool_conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=False)
        self.avg_pool_conv.weight.data.fill_(1.0 / (kernel_size * kernel_size))

    def forward(self, x):
        return self.avg_pool_conv(x)

# 测试
input_tensor = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]).reshape(1, 1, 3, 3)
avg_pool_as_conv = AvgPool2dAsConv2d(2)
result = avg_pool_as_conv(input_tensor)
print(result)


# ### 2. 尝试将最大汇聚层作为卷积层的特殊情况实现。

# In[11]:


import torch
from torch import nn
import torch.nn.functional as F

class MaxPool2dAsConv2d(nn.Module):
    def __init__(self, kernel_size):
        super(MaxPool2dAsConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.max_pool_conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        
        # 设置一个大的负权重和一个大的正偏置
        self.max_pool_conv.weight.data.fill_(-1e6)
        self.max_pool_conv.bias.data.fill_(1e6)

    def forward(self, x):
        # 卷积后应用ReLU激活函数来模拟最大值的选择
        return F.relu(self.max_pool_conv(x))

# 测试
input_tensor = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]).reshape(1, 1, 3, 3)
max_pool_as_conv = MaxPool2dAsConv2d(2)
result = max_pool_as_conv(input_tensor)
print(result)


# ### 3. 汇聚层的计算成本

# 汇聚层的计算成本相对较低。具体计算成本取决于汇聚窗口的大小、步幅和填充。每个输出元素的计算涉及$p_h \times p_w$个操作（最大或平均），总共有 $\lceil (h - p_h + 2 \times pad_h) / s_h \rceil \times \lceil (w - p_w + 2 \times pad_w) / s_w \rceil$ 个输出元素。因此，总操作数为 $c \times \lceil (h - p_h + 2 \times pad_h) / s_h \rceil \times \lceil (w - p_w + 2 \times pad_w) / s_w \rceil \times p_h \times p_w$。

# ### 4. 最大汇聚层和平均汇聚层的工作方式

# 最大汇聚层和平均汇聚层的主要区别在于它们聚合输入数据的方式。最大汇聚层通过选择汇聚窗口中的最大值来保留最强的信号，而平均汇聚层则计算窗口中所有值的平均值，提供平滑的下采样。这两种方法的选择取决于特定应用的需求。

# ### 5. 是否需要最小汇聚层

# 最小汇聚层在实践中较少使用，因为它倾向于捕获背景信息而非突出特征。如果需要，可以通过对输入数据取负值，然后应用最大汇聚层，最
# 
# 后再取负值来模拟最小汇聚层。

# ### 6. 其他汇聚函数

# 除了最大和平均汇聚，也可以考虑使用其他函数，如L2汇聚（平方后取平均，然后开方）或softmax汇聚。这些方法在特定应用中可能有用，但不如最大和平均汇聚流行，主要是因为它们的计算复杂性更高，且在实践中未必总能提供更好的结果。
