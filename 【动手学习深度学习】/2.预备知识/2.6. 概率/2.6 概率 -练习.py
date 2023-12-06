#!/usr/bin/env python
# coding: utf-8

# ## 练习

# 1. 进行$m=500$组实验，每组抽取$n=10$个样本。改变$m$和$n$，观察和分析实验结果。

# In[4]:


import numpy as np
import matplotlib.pyplot as plt

m, n = 500, 10  # 可以修改这些值来观察不同情况
sample_means = np.mean(np.random.normal(size=(m, n)), axis=1)

plt.hist(sample_means, bins=30, edgecolor='black')
plt.xlabel('Sample Mean')
plt.ylabel('Frequency')
plt.title('Distribution of Sample Means')
plt.show()


# 2. 给定两个概率为$P(\mathcal{A})$和$P(\mathcal{B})$的事件，计算$P(\mathcal{A} \cup \mathcal{B})$和$P(\mathcal{A} \cap \mathcal{B})$的上限和下限。（提示：使用[友元图](https://en.wikipedia.org/wiki/Venn_diagram)来展示这些情况。)

# #### 答案
# - \( P(\mathcal{A} \cup \mathcal{B}) \) 的上限是1，下限是 \(\max(P(\mathcal{A}), P(\mathcal{B}))\)，因为事件的概率不能超过1，且至少有一个事件发生的概率不小于单独任一事件发生的概率。
# - \( P(\mathcal{A} \cap \mathcal{B}) \) 的上限是 \(\min(P(\mathcal{A}), P(\mathcal{B}))\)，下限是0，因为两个事件都发生的概率不会超过它们中最不可能发生的事件的概率，且可能完全不发生。

# 3. 假设我们有一系列随机变量，例如$A$、$B$和$C$，其中$B$只依赖于$A$，而$C$只依赖于$B$，能简化联合概率$P(A, B, C)$吗？（提示：这是一个[马尔可夫链](https://en.wikipedia.org/wiki/Markov_chain)。)

# #### 答案
# 根据马尔可夫链的性质，联合概率可以简化为 \( P(A, B, C) = P(A)P(B|A)P(C|B) \)。这是因为每个变量仅依赖于它的前一个变量。

# 4. 在  '2.6.2.6节`中，第一个测试更准确。为什么不运行第一个测试两次，而是同时运行第一个和第二个测试?

# #### 答案
# 运行两个不同的测试（而不是两次相同的测试）可以提供更多的信息和更高的准确性。第二次测试使我们能够对患病的情况获得更高的信心。 尽管第二次检验比第一次检验的准确性要低得多，但它仍然显著提高我们的预测概率。

# In[ ]:




