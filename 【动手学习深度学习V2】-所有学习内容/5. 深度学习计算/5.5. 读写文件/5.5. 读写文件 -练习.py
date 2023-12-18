#!/usr/bin/env python
# coding: utf-8

# # 练习
# 
# 1. 即使不需要将经过训练的模型部署到不同的设备上，存储模型参数还有什么实际的好处？
# 1. 假设我们只想复用网络的一部分，以将其合并到不同的网络架构中。比如想在一个新的网络中使用之前网络的前两层，该怎么做？
# 1. 如何同时保存网络架构和参数？需要对架构加上什么限制？
# 

# ## 1. 即使不需要将经过训练的模型部署到不同的设备上，存储模型参数还有什么实际的好处？

# 存储模型参数，即使不部署到不同设备，有以下好处：
# 
# 中断和恢复训练：在长时间或资源密集型训练过程中，能够在训练中断时保存进度，并在之后从断点继续训练。
# 模型分析和优化：通过保存不同训练阶段的模型，可以分析模型在训练过程中的表现和行为，帮助优化模型架构和参数。
# 模型共享：便于与其他研究者或开发者共享模型，促进合作和知识传播。

# ## 2. 假设我们只想复用网络的一部分，以将其合并到不同的网络架构中。比如想在一个新的网络中使用之前网络的前两层，该怎么做？

# ### 要实现这一点，可以采取以下步骤：
#     1.加载原始模型：首先，加载完整的预训练模型。
#     2.创建新模型：然后，定义一个新的网络架构。
#     3.复制参数：从原始模型中提取前两层的参数，并将它们应用到新模型的对应层上。

# In[9]:


## 代码实现
import torch
from torch import nn
import torch.nn.functional as F

# 定义原始的MLP模型
class OriginalMLP(nn.Module):
    def __init__(self):
        super(OriginalMLP, self).__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

# 定义新的网络模型
class NewNet(nn.Module):
    def __init__(self, original_model):
        super(NewNet, self).__init__()
        self.original_hidden = original_model.hidden
        # 新网络的其他层
        self.new_layer = nn.Linear(256, 20)

    def forward(self, x):
        x = F.relu(self.original_hidden(x))
        x = self.new_layer(x)
        return x

# 实例化原始模型并加载预训练参数
original_model = OriginalMLP()
# 假设有预训练参数
# original_model.load_state_dict(torch.load('path_to_pretrained_params'))

# 实例化新模型
new_net = NewNet(original_model)

# 测试新模型
X_new = torch.randn(size=(2, 20))
Y_new = new_net(X_new)
print(Y_new)


# ## 3. 如何同时保存网络架构和参数？需要对架构加上什么限制？

# In[10]:


# 定义MLP模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

# 实例化并训练模型（这里仅示例，未实际进行训练）
net = MLP()

# 保存模型参数和架构信息
torch.save({'model_state_dict': net.state_dict(),
            'model_class': MLP}, 'model_with_arch.pth')

# 加载模型
saved_info = torch.load('model_with_arch.pth')
model_class = saved_info['model_class']
loaded_model = model_class()
loaded_model.load_state_dict(saved_info['model_state_dict'])

# 测试加载的模型
loaded_model.eval()
X = torch.randn(size=(2, 20))
Y_loaded = loaded_model(X)
print(Y_loaded)


# In[ ]:




