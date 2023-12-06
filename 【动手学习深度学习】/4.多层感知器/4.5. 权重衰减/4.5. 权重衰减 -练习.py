#!/usr/bin/env python
# coding: utf-8

# # 权重衰减

# In[52]:


get_ipython().run_line_magic('matplotlib', 'inline')
import torch
from torch import nn
from d2l import torch as d2l


# In[53]:


# 数据生成
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)


# In[54]:


# 初始化参数
def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]


# In[55]:


# L2 范数惩罚
def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2


# In[56]:


# 训练函数
def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
    train_loss = d2l.evaluate_loss(net, train_iter, loss)
    test_loss = d2l.evaluate_loss(net, test_iter, loss)
    return train_loss, test_loss


# In[57]:


# 练习1：绘制训练和测试精度关于λ的函数
lambdas = [0, 1, 2, 3, 4, 5]
train_losses, test_losses = [], []
for lambd in lambdas:
    train_loss, test_loss = train(lambd)
    train_losses.append(train_loss)
    test_losses.append(test_loss)

d2l.plot(lambdas, [train_losses, test_losses], 'lambda', 'loss', 
         legend=['train', 'test'], xscale='log')


# ## 练习2：使用验证集找到最佳λ值

# In[58]:


# 分割验证数据集
n_valid = 5  # 验证集的大小
valid_features = features[:n_valid]
valid_labels = labels[:n_valid]
valid_data = (valid_features, valid_labels)
valid_iter = d2l.load_array(valid_data, batch_size, is_train=False)

# 剩余的作为新的训练数据集
train_features_new = features[n_valid:]
train_labels_new = labels[n_valid:]
train_data_new = (train_features_new, train_labels_new)
train_iter_new = d2l.load_array(train_data_new, batch_size)


# In[59]:


def train_with_valid(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    for epoch in range(num_epochs):
        for X, y in train_iter_new:  # 使用新的训练数据集
            l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
    valid_loss = d2l.evaluate_loss(net, valid_iter, loss)  # 在验证集上评估
    return valid_loss

# 测试不同的λ值
lambdas = [0, 0.01, 0.1, 1, 10, 100]
valid_losses = []
for lambd in lambdas:
    valid_loss = train_with_valid(lambd)
    valid_losses.append(valid_loss)

# 找出最佳的λ值
best_lambd = lambdas[valid_losses.index(min(valid_losses))]
print("Best lambda:", best_lambd)


# ### 练习3：L1正则化的更新方程
# 
# L1正则化在梯度下降中的应用会导致权重向量更稀疏。这是因为L1正则化在优化过程中会倾向于将权重推向0。修改后的`l1_penalty`函数和相应的训练函数如下：
# 
# ```python
# def l1_penalty(w):
#     return torch.sum(torch.abs(w))
# 
# def train_l1(lambd):
#     w, b = init_params()
#     net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
#     num_epochs, lr = 100, 0.003
#     for epoch in range(num_epochs):
#         for X, y in train_iter:
#             l = loss(net(X), y) + lambd * l1_penalty(w)
#             l.sum().backward()
#             d2l.sgd([w, b], lr, batch_size)
#     print('w的L1范数是：', torch
# 
# .norm(w, 1).item())
# ```
# 
# ### 练习4：矩阵形式的Frobenius范数
# 
# Frobenius范数是一个矩阵范数，用于测量矩阵的大小。对于矩阵\(A\)，其Frobenius范数定义为\(\sqrt{\sum_{i,j} A_{ij}^2}\)。这可以看作是矩阵元素平方的总和的平方根，类似于向量的L2范数。在代码中，你可以用`torch.norm`来计算一个矩阵的Frobenius范数：
# 
# ```python
# A = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
# frobenius_norm = torch.norm(A)
# print('Frobenius norm of A:', frobenius_norm.item())
# ```
# 
# ### 练习5：处理过拟合的其他方法
# 
# 除了权重衰减、增加训练数据和使用合适复杂度的模型，还有几种其他方法可以处理过拟合：
# 
# 1. **Dropout**：在训练过程中随机地“关闭”一些神经元，这可以使模型对少量特定的数据点不那么敏感。
# 2. **早停（Early Stopping）**：当验证集上的性能不再提升时停止训练，防止模型在训练数据上过度拟合。
# 3. **批量归一化（Batch Normalization）**：通过调整层的输入使其更加标准化，有助于缓解内部协变量偏移问题。
# 4. **数据增强（Data Augmentation）**：在训练过程中对输入数据进行随机的变换，从而增加数据的多样性。
# 
# 这些方法可以单独使用，也可以组合使用，以提高模型的泛化能力。
# 
# ### 练习6：正则化的贝叶斯解释
# 
# 在贝叶斯统计框架中，正则化可以被解释为对模型参数的先验概率分布。例如，L2正则化对应于对参数应用高斯先验，而L1正则化对应于拉普拉斯先验。这些先验反映了我们对参数值的信念，例如，高斯先验假设大多数权重应该接近0。
# 
# 贝叶斯更新公式为 \(P(w | x) \propto P(x | w) P(w)\)，其中 \(P(w)\) 是先验，\(P(x | w)\) 是似然，\(P(w | x)\) 是后验。正则化项对应于先验 \(P(w)\)。例如，对于L2正则化，先验 \(P(w)\) 可以是以0为均值的高斯分布。通过这种方式，正则化可以被视为在优化过程中将先验知识合并到模型中。

# In[ ]:





# In[ ]:




