#!/usr/bin/env python
# coding: utf-8

# ## 练习
# 
# 1. 如果不同方向使用不同数量的隐藏单位，$\mathbf{H_t}$的形状会发生怎样的变化？
# 1. 设计一个具有多个隐藏层的双向循环神经网络。
# 1. 在自然语言中一词多义很常见。例如，“bank”一词在不同的上下文“i went to the bank to deposit cash”和“i went to the bank to sit down”中有不同的含义。如何设计一个神经网络模型，使其在给定上下文序列和单词的情况下，返回该单词在此上下文中的向量表示？哪种类型的神经网络架构更适合处理一词多义？
# 

# In[5]:


import torch
from torch import nn
import torch.utils.data as data

# 定义一个简化的双向LSTM模型，使用不同数量的隐藏单元
class SimpleBiLSTM(nn.Module):
    def __init__(self, vocab_size, num_hiddens_forward, num_hiddens_backward, num_layers):
        super(SimpleBiLSTM, self).__init__()
        # 分别定义前向和后向的LSTM层
        self.lstm_forward = nn.LSTM(vocab_size, num_hiddens_forward, num_layers, batch_first=True)
        self.lstm_backward = nn.LSTM(vocab_size, num_hiddens_backward, num_layers, batch_first=True)
        # 定义全连接层
        self.fc = nn.Linear(num_hiddens_forward + num_hiddens_backward, vocab_size)

    def forward(self, x):
        # 分别处理前向和后向的数据
        out_forward, _ = self.lstm_forward(x)
        out_backward, _ = self.lstm_backward(torch.flip(x, [1]))
        # 合并前向和后向的输出
        out = torch.cat((out_forward, out_backward), -1)
        # 通过全连接层
        out = self.fc(out)
        return out

# 设置模型参数
vocab_size = 1024  # 假设的词汇量大小
num_hiddens_forward = 128
num_hiddens_backward = 256
num_layers = 2

# 创建模型实例
model = SimpleBiLSTM(vocab_size, num_hiddens_forward, num_hiddens_backward, num_layers)

# 创建一个随机输入张量来测试模型
test_input = torch.rand(32, 35, vocab_size)  # 批量大小为32，时间步为35
test_output = model(test_input)

test_output.shape  # 检查输出张量的形状


# In[6]:


## 练习2
import torch
from torch import nn

class MultiLayerBiLSTM(nn.Module):
    def __init__(self, vocab_size, num_hiddens, num_layers):
        super(MultiLayerBiLSTM, self).__init__()
        # 定义多层双向LSTM
        self.lstm = nn.LSTM(vocab_size, num_hiddens, num_layers, bidirectional=True)
        # 定义输出层
        self.fc = nn.Linear(num_hiddens * 2, vocab_size)  # 乘以2是因为双向

    def forward(self, x):
        # LSTM层
        output, _ = self.lstm(x)
        # 通过全连接层
        out = self.fc(output)
        return out

# 参数设置
vocab_size = 1024  # 假设的词汇量大小
num_hiddens = 256
num_layers = 3  # 多层

# 创建模型实例
model = MultiLayerBiLSTM(vocab_size, num_hiddens, num_layers)

# 测试模型
test_input = torch.rand(32, 35, vocab_size)  # 批量大小为32，时间步为35
test_output = model(test_input)

test_output.shape  # 输出形状


# In[8]:


## 练习3
from transformers import BertTokenizer, BertModel
import torch

# 初始化BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 示例句子
sentence = "I went to the bank to deposit cash."

# 对句子进行分词，并获取各个词的嵌入
inputs = tokenizer(sentence, return_tensors="pt")
outputs = model(**inputs)

# outputs中包含了句子中每个词的上下文相关嵌入
word_embeddings = outputs.last_hidden_state


# In[ ]:




