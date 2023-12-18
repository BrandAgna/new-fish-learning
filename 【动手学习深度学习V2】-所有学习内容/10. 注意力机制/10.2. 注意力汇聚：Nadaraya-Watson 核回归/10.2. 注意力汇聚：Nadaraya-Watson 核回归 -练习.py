#!/usr/bin/env python
# coding: utf-8

# ## 练习
# 
# 1. 增加训练数据的样本数量，能否得到更好的非参数的Nadaraya-Watson核回归模型？
# 1. 在带参数的注意力汇聚的实验中学习得到的参数$w$的价值是什么？为什么在可视化注意力权重时，它会使加权区域更加尖锐？
# 1. 如何将超参数添加到非参数的Nadaraya-Watson核回归中以实现更好地预测结果？
# 1. 为本节的核回归设计一个新的带参数的注意力汇聚模型。训练这个新模型并可视化其注意力权重。

# ## 练习1
# 在非参数的Nadaraya-Watson核回归模型中，模型的预测是基于训练数据点及其与查询点（即测试样本）之间的关系计算得出的。该模型不假设数据的任何固定参数形式，而是直接使用训练数据来估计函数形态。在这种设置下，增加训练样本数量通常会提高模型的性能，原因如下：
# 
# 1. **更丰富的数据覆盖**：增加样本数量可以提高训练数据对整个数据分布的覆盖度。在非参数模型中，数据覆盖的广泛性是非常重要的，因为模型的预测完全依赖于训练数据。更广泛的覆盖意味着模型能够更好地了解和适应数据的整体分布。
# 
# 2. **改善局部估计**：Nadaraya-Watson模型特别依赖于邻近训练点来进行局部估计。增加样本数量会增加邻近点的数量，从而提供更准确的局部估计，尤其是在数据分布较为复杂或具有较大变异性的区域。
# 
# 3. **减少过度拟合的风险**：虽然非参数模型不太容易过拟合（因为它们没有固定的参数形式），但当训练数据很少时，模型可能对这些少量数据点产生过度敏感。增加数据量可以帮助模型平滑出这种过度敏感性，提供更一般化的预测。
# 
# 然而，值得注意的是，非参数模型的计算成本通常随着数据量的增加而显著增加。特别是在Nadaraya-Watson模型中，计算每个查询点的输出需要考虑其与所有训练点之间的关系。因此，对于非常大的数据集，这种模型可能会变得非常低效。此外，如果新增的数据质量不高（例如，包含很多噪声或异常值），它们可能不会带来预期的性能提升，甚至可能损害模型的性能。
# 
# 综上所述，增加训练样本数量通常可以提高Nadaraya-Watson核回归模型的性能，但同时也要考虑到计算效率和数据质量的影响。

# ## 练习2
#    在带参数的注意力汇聚中，参数$w$控制着注意力权重的分布。这个参数影响着高斯核的宽度，即决定了注意力聚焦的紧密程度。一个较小的$w$值会导致更尖锐的注意力分布，这意味着模型在做出预测时更加专注于邻近的少数几个点。这有助于模型在有噪声的数据中更加关注于那些与查询点最接近的训练样本。

# ## 练习3
# 
#    一个常见的方法是引入一个可调整的带宽超参数（类似于$w$），这个超参数决定了高斯核的宽度。通过调整带宽，可以控制模型对训练数据点的敏感度。较小的带宽使模型更专注于接近查询点的训练样本，而较大的带宽使模型在预测时考虑更广泛的数据点。选择合适的带宽是关键，需要基于验证集性能进行调整。

# In[12]:


## 练习4
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

class ParametricAttention(nn.Module):
    def __init__(self):
        super(ParametricAttention, self).__init__()
        self.W = nn.Parameter(torch.rand(1))

    def forward(self, queries, keys, values):
        queries = queries.unsqueeze(1)
        self.attention_weights = nn.functional.softmax(-self.W * (queries - keys) ** 2, dim=1)
        return (self.attention_weights * values).sum(dim=1), self.attention_weights

x = torch.linspace(-1, 1, 10).reshape(-1, 1)
y = x ** 2

model = ParametricAttention()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output, _ = model(x, x, y)
    loss = loss_fn(output, y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

with torch.no_grad():
    _, attention_weights = model(x, x, y)
    # 提取对角线上的注意力权重
    attention_weights = attention_weights.squeeze().diag().numpy()

x_plot = x.squeeze().numpy()

plt.scatter(x_plot, y.numpy(), label='Data points')
plt.scatter(x_plot, attention_weights, color='r', label='Attention weights')
plt.xlabel('x')
plt.ylabel('y/weights')
plt.legend()
plt.show()


# In[ ]:




