#!/usr/bin/env python
# coding: utf-8

# ## 练习
# 
# 1. 在`load_data_nmt`函数中尝试不同的`num_examples`参数值。这对源语言和目标语言的词表大小有何影响？
# 1. 某些语言（例如中文和日语）的文本没有单词边界指示符（例如空格）。对于这种情况，单词级词元化仍然是个好主意吗？为什么？

# In[6]:


import os
import torch
from d2l import torch as d2l

def tokenize_nmt(text, num_examples=None):
    """词元化“英语－法语”数据数据集"""
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target

source, target = tokenize_nmt(text)
source[:6], target[:6]

def read_data_nmt():
    """载入“英语－法语”数据集"""
    data_dir = d2l.download_extract('fra-eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r',
             encoding='utf-8') as f:
        return f.read()

raw_text = read_data_nmt()
print(raw_text[:75])

def preprocess_nmt(text):
    """预处理“英语－法语”数据集"""
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # 使用空格替换不间断空格
    # 使用小写字母替换大写字母
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 在单词和标点符号之间插入空格
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)

text = preprocess_nmt(raw_text)
print(text[:80])

def load_data_nmt(batch_size, num_steps, num_examples=600):
    """返回翻译数据集的迭代器和词表"""
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)
    src_vocab = d2l.Vocab(source, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = d2l.Vocab(target, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = d2l.load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab

src_vocab = d2l.Vocab(source, min_freq=2,
                      reserved_tokens=['<pad>', '<bos>', '<eos>'])
len(src_vocab)

def truncate_pad(line, num_steps, padding_token):
    """截断或填充文本序列"""
    if len(line) > num_steps:
        return line[:num_steps]  # 截断
    return line + [padding_token] * (num_steps - len(line))  # 填充

truncate_pad(src_vocab[source[0]], 10, src_vocab['<pad>'])

def build_array_nmt(lines, vocab, num_steps):
    """将机器翻译的文本序列转换成小批量"""
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = torch.tensor([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len

# 定义一个函数来观察不同num_examples值对词汇表大小的影响
def explore_vocab_size(num_examples_list, num_steps=8):
    vocab_sizes = {}
    for num_examples in num_examples_list:
        _, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=num_steps, num_examples=num_examples)
        vocab_sizes[num_examples] = (len(src_vocab), len(tgt_vocab))
    return vocab_sizes

# 不同的num_examples值
num_examples_list = [100, 300, 600, 1200, 2400]

# 观察词汇表大小的变化
vocab_size_changes = explore_vocab_size(num_examples_list)
vocab_size_changes


# ## 练习2
# 在处理像中文和日语这样的语言时，单词级别的词元化（基于空格的分词）通常不是一个有效的策略，因为这些语言的写作系统不使用空格来分隔单词。这意味着，如果我们简单地按空格来分词，将无法正确地分离单词。
# 
# 对于这些语言，更常用的方法是：
# 
# 1. **字符级词元化**：一种策略是将每个字符视为一个独立的词元。这种方法在某些情况下可能有效，但它忽略了字符之间的组合，这些组合在很多情况下才构成有意义的单词或短语。
# 
# 2. **子词词元化**：这是一种更为先进的策略，其中使用特定的算法（如Byte Pair Encoding (BPE)、WordPiece等）来识别和使用语料库中的常见字符组合。这种方法能够有效地捕捉单词内部的结构，同时保持对未知或罕见单词的灵活性。
# 
# 3. **基于词典的词元化**：对于中文和日语，还有一种方法是使用词典来进行分词。这种方法依赖于大型的、经过细致编辑的词典，能够识别出各种单词和短语。词典分词通常能够获得较高的准确度，但它的缺点在于处理未知单词的能力有限。
# 
# 综上所述，对于中文和日语这样的语言，单词级词元化通常不是最佳选择。更适合的方法是子词词元化或基于词典的词元化，这些方法能够更好地处理没有明显单词边界的语言。

# In[ ]:




