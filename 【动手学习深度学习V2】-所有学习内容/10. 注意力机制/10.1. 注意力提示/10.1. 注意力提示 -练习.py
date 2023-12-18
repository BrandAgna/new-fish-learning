#!/usr/bin/env python
# coding: utf-8

# ## 练习
# 
# 1. 在机器翻译中通过解码序列词元时，其自主性提示可能是什么？非自主性提示和感官输入又是什么？
# 1. 随机生成一个$10 \times 10$矩阵并使用`softmax`运算来确保每行都是有效的概率分布，然后可视化输出注意力权重。
# 

# ## 练习1
#    - **自主性提示（Autoregressive Prompt）**：在机器翻译中，自主性提示是指在生成翻译序列时，当前步骤的输出依赖于前一步骤的输出。例如，在生成一个句子的过程中，下一个词元的选择依赖于前一个或前几个已经生成的词元。这种依赖性使得生成过程是自回归的，即每一步的输出都是前一步输出的函数。
# 
#    - **非自主性提示（Non-autoregressive Prompt）**：非自主性提示指的是在生成过程中，当前步骤的输出不直接依赖于前一步骤的具体输出。在非自主性机器翻译模型中，整个句子或一大部分句子可以同时生成，而不是逐词生成。这种方法通常速度更快，但可能在准确性和流畅性上存在挑战。
# 
#    - **感官输入（Sensory Input）**：在机器翻译的上下文中，感官输入通常指的是模型接收到的原始输入数据，例如源语言中的句子。这些输入是模型用来生成翻译的基础，它们通常被编码为一系列的词元或词向量，供模型进一步处理。

# In[3]:


import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

# 生成一个随机的 10x10 矩阵
matrix = torch.rand(10, 10)

# 使用 softmax 转换成概率分布
attention_weights = F.softmax(matrix, dim=1)

# 可视化注意力权重
plt.figure(figsize=(8, 6))
sns.heatmap(attention_weights.numpy(), annot=True, cmap='Blues')
plt.xlabel('Attention Weight')
plt.ylabel('Token')
plt.show()


# In[ ]:




