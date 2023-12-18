#!/usr/bin/env python
# coding: utf-8

# ## 练习
# 
# 1. 词元化是一个关键的预处理步骤，它因语言而异。尝试找到另外三种常用的词元化文本的方法。
# 1. 在本节的实验中，将文本词元为单词和更改`Vocab`实例的`min_freq`参数。这对词表大小有何影响？
# 

# ### 练习 1: 常用的词元化方法
# 
# 词元化是将文本分割为更小单元（词元）的过程。不同语言和应用场景可能需要不同的词元化方法。除了按单词和字符分割外，以下是三种常用的词元化方法：
# 
# 1. **基于子词（Subword）的词元化**：这种方法将词分割为更小的有意义的单元，例如，单词"better"可以被分割为"bet"和"ter"。这对于处理生僻词和构建更小的词表非常有效。常用的子词词元化算法包括Byte-Pair Encoding (BPE)和WordPiece。
# 
# 2. **基于n-gram的词元化**：n-gram是文本中连续的n个项。例如，对于句子"The cat sat on the mat"，它的2-gram（或称bigram）序列为["The cat", "cat sat", "sat on", "on the", "the mat"]。n-gram词元化有助于捕捉局部上下文信息。
# 
# 3. **基于正则表达式的词元化**：在这种方法中，使用正则表达式来定义词元的边界。这对于特定类型的文本（如程序代码或具有复杂结构的文本）非常有用。

# In[2]:


import collections
import re
from d2l import torch as d2l

# 加载时光机器数据集
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():  # @save
    """将时间机器数据集加载到文本行的列表中"""
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

lines = read_time_machine()
print(f'# 文本总行数: {len(lines)}')
print(lines[0])
print(lines[10])

# 词元化函数
def tokenize(lines, token='word'):  # @save
    """将文本行拆分为单词或字符词元"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)

# 使用单词级词元化
tokens = tokenize(lines, 'word')
for i in range(11):
    print(tokens[i])

# 词频统计函数
def count_corpus(tokens):  # @save
    """统计词元的频率"""
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

# 词表类
class Vocab:  # @save
    """文本词表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

# 创建词表，设置min_freq参数
min_freq = 3
vocab = Vocab(tokens, min_freq)
print(f"词表大小: {len(vocab)}")
print(list(vocab.token_to_idx.items())[:10])

# 显示一些文本行及其对应的索引
for i in [0, 10]:
    print('文本:', tokens[i])
    print('索引:', vocab[tokens[i]])


# In[ ]:




