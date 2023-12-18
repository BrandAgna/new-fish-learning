#!/usr/bin/env python
# coding: utf-8

# ## 练习：
# ### 创建包含更多行和列的原始数据集。
# ### 删除缺失值最多的列。
# ### 将预处理后的数据集转换为张量格式

# # 创建更大的数据集：

# In[14]:


import os


# In[15]:


os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')


# In[16]:


import pandas as pd
import numpy as np

data = pd.read_csv(data_file)

# 添加更多的行和列
# 为了演示，我们使用随机数填充新列
additional_data = pd.DataFrame(np.random.rand(10, 5), columns=['A', 'B', 'C', 'D', 'E'])
data = pd.concat([data, additional_data], axis=0)

print(data)


# # 删除缺失值最多的列：

# In[19]:


# 计算每列的缺失值数量
missing_counts = data.isnull().sum()

# 找到缺失值最多的列
max_missing = missing_counts.idxmax()

# 删除该列
data = data.drop(max_missing, axis=1)

print(data)


# # 将预处理后的数据集转换为张量：

# In[22]:


import torch

# 假设所有剩余的数据都是数值型，直接转换为张量
tensor_data = torch.tensor(data.values)

tensor_data

