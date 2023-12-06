#!/usr/bin/env python
# coding: utf-8

# ### 练习问题 1
# 假设我们有一些数据 \( x_1, \ldots, x_n \in \mathbb{R} \)。我们的目标是找到一个常数 \( b \)，使得最小化 \( \sum_i (x_i - b)^2 \)。
# 
# #### 1.A. 找到最优值 \( b \) 的解析解
# 为了找到最小化平方误差的 \( b \)，我们需要对 \( b \) 求导，并将导数设为0。求解得到的方程会给出 \( b \) 的最优值。
# 
# #### 解析解
# 对于函数 \( f(b) = \sum_i (x_i - b)^2 \)，它的导数是 \( f'(b) = -2 \sum_i (x_i - b) \)。将导数设为0，得到：
# \[ -2 \sum_i (x_i - b) = 0 \]
# \[ \sum_i x_i - nb = 0 \]
# \[ b = \frac{\sum_i x_i}{n} \]
# 
# 所以，\( b \) 的最优值是 \( x_i \) 的平均值。
# 
# #### 1.B. 这个问题及其解与正态分布有什么关系?
# 最小化平方误差等价于在正态分布假设下的最大似然估计。如果我们假设 \( x_i \) 的观测值是从真实值 \( b \) 及加上正态分布的噪声得到的，那么最大化观测数据的似然等价于最小化误差平方和。这是因为正态分布的对数似然包含了一个 \( -(x_i - b)^2 \) 的项。
# 
# ### 练习问题 2
# 推导出使用平方误差的线性回归优化问题的解析解。
# 
# #### 2.A. 用矩阵和向量表示法写出优化问题
# 如果我们有一个数据矩阵 \( \mathbf{X} \) 和目标值向量 \( \mathbf{y} \)，线性回归模型可以表示为 \( \hat{\mathbf{y}} = \mathbf{X}\mathbf{w} \)，其中 \( \mathbf{w} \) 是权重向量。优化问题是最小化 \( \|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2 \)。
# 
# #### 2.B. 计算损失对 \( w \) 的梯度
# 损失函数 \( L(\mathbf{w}) = \|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2 \) 对 \( \mathbf{w} \) 的梯度是 \( \nabla_{\mathbf{w}} L(\mathbf{w}) = -2\mathbf{X}^\top (\mathbf{y} - \mathbf{X}\mathbf{w}) \)。
# 
# #### 2.C. 通过将梯度设为0求解矩阵方程
# 将梯度设为0得到 \( \mathbf{X}^\top \mathbf{X}\mathbf{w} = \mathbf{X}^\top \mathbf{y} \)。如果 \( \mathbf{X}^\top \mathbf{X} \) 是可逆的，我们可以左乘它的逆矩阵得到 \( \mathbf{w} = (\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top \mathbf{y} \)。
# 
# #### 2.D. 什么时候可能比使用随机梯度下降更好？
# 解析解在数据集较小且 \( \mathbf{X}^\top \mathbf{X} \) 易于计算和可逆时效果较好。然而，当数据集很大或者 \( \mathbf{X
# 
# }^\top \mathbf{X} \) 不可逆（例如，特征之间高度相关）时，解析解要么计算量太大，要么不存在。在这些情况下，使用随机梯度下降或其他优化算法可能更好。
# 
# ### 练习问题 3
# 假定控制附加噪声 \( \epsilon \) 的噪声模型是指数分布。
# 
# #### 3.A. 写出模型 \( -\log P(\mathbf{y} | \mathbf{X}) \) 下数据的负对数似然
# 如果噪声 \( \epsilon \) 遵循指数分布，我们有 \( P(\epsilon) = \frac{1}{2} \exp(-|\epsilon|) \)。负对数似然函数是 \( -\log P(\mathbf{y} | \mathbf{X}) = -\sum_i \log P(y_i | \mathbf{x}_i) \)。由于 \( y_i = \mathbf{w}^\top \mathbf{x}_i + \epsilon \)，我们可以将 \( \epsilon \) 替换为 \( y_i - \mathbf{w}^\top \mathbf{x}_i \)，得到负对数似然函数 \( L(\mathbf{w}) = \sum_i |\mathbf{w}^\top \mathbf{x}_i - y_i| \)。
# 
# #### 3.B. 请试着写出解析解
# 指数分布导致的损失函数是绝对误差，这与平方误差不同，不会有简单的闭合形式解析解。这是一个L1范数优化问题，可以通过线性规划方法求解。
# 
# #### 3.C. 提出一种随机梯度下降算法来解决这个问题
# 我们可以使用随机梯度下降算法最小化负对数似然函数 \( L(\mathbf{w}) \)。然而，由于绝对值函数在0点不可导，我们需要为 \( \mathbf{w}^\top \mathbf{x}_i = y_i \) 的情况定义一个子梯度。子梯度可以是区间 \([-1, 1]\) 内的任意值。

# In[3]:


import numpy as np

# 示例数据，您需要根据实际情况替换 [...] 为具体的数值
X = np.array([[1, 2], [3, 4]])  # 用实际的输入数据替换
y = np.array([1, 2])            # 用实际的输出数据替换

# 初始化权重为零
w = np.zeros(X.shape[1])

# 学习率
lr = 0.01

# 随机梯度下降
for i in range(1000):  # 进行1000次迭代
    # 随机选取一个样本
    idx = np.random.randint(X.shape[0])
    x_i, y_i = X[idx], y[idx]
    # 计算预测误差
    error = np.dot(w, x_i) - y_i
    # 计算梯度
    grad = np.sign(error) * x_i
    # 更新权重
    w -= lr * grad

# 打印最终权重
print(w)

