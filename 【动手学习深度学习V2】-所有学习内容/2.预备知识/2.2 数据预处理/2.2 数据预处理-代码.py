#!/usr/bin/env python
# coding: utf-8

# ## 读取数据集

# In[24]:


import os


# In[25]:


os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')


# In[26]:


import pandas as pd
data = pd.read_csv(data_file)
print(data)


# ## 处理缺失值

# In[27]:


# 分离数值列和非数值列
numeric_inputs = inputs.select_dtypes(include=['number'])
non_numeric_inputs = inputs.select_dtypes(exclude=['number'])

# 只对数值列填充均值
numeric_inputs = numeric_inputs.fillna(numeric_inputs.mean())

# 将处理后的数值列和非数值列重新合并
inputs = pd.concat([numeric_inputs, non_numeric_inputs], axis=1)
print(inputs)


# In[28]:


inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)


# ## 转换为张量格式

# In[29]:


import torch

X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(outputs.to_numpy(dtype=float))
X, y

