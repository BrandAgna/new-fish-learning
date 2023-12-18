#!/usr/bin/env python
# coding: utf-8

# ## 练习
# 
# 1. 打印层的尺寸时，我们只看到8个结果，而不是11个结果。剩余的3层信息去哪了？
# 1. 与AlexNet相比，VGG的计算要慢得多，而且它还需要更多的显存。分析出现这种情况的原因。
# 1. 尝试将Fashion-MNIST数据集图像的高度和宽度从224改为96。这对实验有什么影响？
# 1. 请参考VGG论文 :cite:`Simonyan.Zisserman.2014`中的表1构建其他常见模型，如VGG-16或VGG-19。
# 

# ### 练习 1 - 层的尺寸输出数量
# 当你打印VGG网络中每层的输出尺寸时，你可能注意到只看到了8个输出，而非预期的11个。剩余的3层信息去哪了？：
# 
# 1. **汇聚层的省略**: `nn.MaxPool2d`汇聚层在`vgg_block`中被添加，但在打印层的尺寸时并没有单独被视为一个层。在打印时，汇聚层的输出被包含在了前面的卷积层输出中。
# 2. **顺序模型的特性**: 由于将`vgg_block`的输出作为一个整体（`nn.Sequential`对象），在打印时它被视为一个单一的层。因此，尽管`vgg_block`中可能有多个卷积层，它们的输出只显示为一个输出。
# 这导致了在打印网络结构时，只展示了较少的层输出结果。

# ### 练习 2 - 与AlexNet相比，VGG的计算要慢得多，而且它还需要更多的显存。分析出现这种情况的原因。
# VGG网络比AlexNet慢且需要更多的显存，主要是因为：
# 1. **更深的网络结构**: VGG网络有更多的卷积层，这增加了网络的深度。更深的网络需要更多的计算资源和时间来进行前向和后向传播。
# 2. **更多的参数**: VGG网络的每个卷积层通常有更多的过滤器，并且网络中有更多的全连接层参数。这导致了显著增加的参数数量，从而需要更多的显存来存储这些参数和中间的激活值。
# 3. **更大的特征图尺寸**: 在VGG的早期层中，由于较小的汇聚层窗口和较大的输入尺寸，特征图（feature maps）的尺寸相对较大。这些大尺寸的特征图需要更多的计算资源和显存。

# ### 练习 3 - 尝试将Fashion-MNIST数据集图像的高度和宽度从224改为96。这对实验有什么影响？

# In[9]:


import torch
from torch import nn
from d2l import torch as d2l

def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    # 根据最后一个卷积层的输出调整全连接层的输入尺寸
    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        nn.Linear(128 * 3 * 3, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))

conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)

# 训练参数设置
lr, num_epochs, batch_size = 0.05, 10, 128

# 使用 96 x 96 尺寸的图像进行训练
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())


# ### 练习 4 - 请参考VGG论文 :cite:`Simonyan.Zisserman.2014`中的表1构建其他常见模型，如VGG-16或VGG-19。

# In[ ]:


import torch
from torch import nn
from d2l import torch as d2l

def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

def vgg(conv_arch, fc_features):
    conv_blks = []
    in_channels = 1
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        nn.Linear(fc_features, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))

def find_fc_features(conv_arch, input_shape=(1, 1, 224, 224)):
    in_channels = 1  # 初始输入通道数为1
    with torch.no_grad():
        net = nn.Sequential(*[vgg_block(num_convs, in_channels if i == 0 else conv_arch[i-1][1], out_channels)
                              for i, (num_convs, out_channels) in enumerate(conv_arch)],
                            nn.Flatten())
        return net(torch.randn(*input_shape)).shape[1]

# VGG-16 架构和训练
vgg16_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
fc_features_vgg16 = find_fc_features(vgg16_arch)
net_vgg16 = vgg(vgg16_arch, fc_features_vgg16)
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net_vgg16, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

# VGG-19 架构和训练
vgg19_arch = ((1, 64), (1, 128), (4, 256), (4, 512), (4, 512))
fc_features_vgg19 = find_fc_features(vgg19_arch)
net_vgg19 = vgg(vgg19_arch, fc_features_vgg19)
d2l.train_ch6(net_vgg19, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())


# In[ ]:




