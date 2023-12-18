#!/usr/bin/env python
# coding: utf-8

# ## 练习
# 
# 1. 分别可视化这个实验中的多个头的注意力权重。
# 1. 假设有一个完成训练的基于多头注意力的模型，现在希望修剪最不重要的注意力头以提高预测速度。如何设计实验来衡量注意力头的重要性呢？

# In[6]:


import math
import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt
import seaborn as sns

class DotProductAttention(nn.Module):
    """缩放点积注意力"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, valid_lens=None):
        d = query.shape[-1]
        scores = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = d2l.masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), value)

class MultiHeadAttention(nn.Module):
    """多头注意力"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # 对查询、键、值进行线性变换并分头
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)

        # 缩放点积注意力
        output = self.attention(queries, keys, values, valid_lens)

        # 存储每个头的注意力权重
        self.head_attention_weights = self.attention.attention_weights
        self.head_attention_weights = self.head_attention_weights.view(batch_size, self.num_heads, queries.shape[1], keys.shape[1])

        # Concatenate attention output for each head
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)

def transpose_qkv(X, num_heads):
    # 将最后一个维度分割成(num_heads, -1)，然后交换中间两个维度
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(-1, X.shape[2], X.shape[3])

def transpose_output(X, num_heads):
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)

def plot_attention_weights(attention_weights, tokens, sentence_length, num_heads, figsize=(15, 3)):
    fig, axes = plt.subplots(1, num_heads, figsize=figsize)
    for i in range(num_heads):
        sns.heatmap(attention_weights[:, i, :sentence_length, :sentence_length][0].detach().numpy(),
                    annot=True, ax=axes[i], fmt='.2f', cbar=False, cmap='viridis')
        axes[i].set_title(f'Head {i+1}')
        axes[i].set_xlabel('Key positions')
        axes[i].set_ylabel('Query positions')
    plt.show()

# 创建实例并应用模型
num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens, num_hiddens, num_heads, 0.5)
attention.eval()

# 创建样例数据
batch_size, num_queries, num_kvpairs, valid_lens = 2, 4, 6, torch.tensor([3, 2])
X = torch.ones((batch_size, num_queries, num_hiddens))
Y = torch.ones((batch_size, num_kvpairs, num_hiddens))

# 应用多头注意力机制
output = attention(X, Y, Y, valid_lens)

# 可视化每个头的注意力权重
for i in range(num_heads):
    plot_attention_weights(attention.head_attention_weights, ["this", "is", "a", "test"], num_queries, num_heads)


# ## 练习2
# 衡量注意力头的重要性并修剪不重要的头以提高模型预测速度是一个复杂且有趣的问题。以下是一个可能的实验设计方法来衡量注意力头的重要性：
# 
# 1. **基线性能评估**:
#    - 在修剪前，评估模型在验证集或测试集上的性能。记录关键性能指标，如准确率、BLEU分数（对于翻译任务）或其他相关指标。
# 
# 2. **逐个移除注意力头**:
#    - 逐个移除每个注意力头，每次移除一个并在同一数据集上重新评估模型性能。注意，在移除注意力头时，相应的线性变换（例如，对应于该头的`W_q`、`W_k`、`W_v`）也应被移除或忽略。
#    - 记录每次移除一个头后的模型性能，并与基线性能进行比较。
# 
# 3. **性能变化分析**:
#    - 分析每次移除头后性能的变化。一个头的重要性可以根据其被移除后对模型性能的影响程度来衡量。如果移除某个头后性能下降显著，则该头较为重要；如果性能影响很小或没有变化，那么这个头可能不那么重要。
# 
# 4. **多头联合分析**:
#    - 除了逐个测试每个头的重要性之外，还可以尝试组合不同的头并评估性能。这是因为头之间可能存在交互作用，某些头的组合可能比单独考虑每个头更重要。
# 
# 5. **确定修剪策略**:
#    - 基于上述实验结果，确定一种修剪策略。这可能涉及到移除一个或多个影响性能最小的头。修剪策略应该在尽可能少影响性能的同时最大化预测速度的提升。
# 
# 6. **最终模型评估**:
#    - 在确定了修剪策略后，对修剪后的模型进行最终评估，以确保其性能仍然符合要求。
# 
# 7. **注意实验的随机性**:
#    - 在这类实验中，确保结果的一致性很重要。可能需要多次运行实验并取平均值，以减少随机性对结果的影响。
# 
# 通过上述步骤，可以相对系统地评估每个注意力头的重要性，并据此作出修剪决策。这种方法既可以揭示每个头的独立贡献，也可以通过组合不同的头来理解它们的联合作用。

# In[ ]:




