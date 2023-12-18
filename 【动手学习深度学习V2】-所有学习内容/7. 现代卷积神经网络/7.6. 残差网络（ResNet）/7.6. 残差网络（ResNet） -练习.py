#!/usr/bin/env python
# coding: utf-8

# ## 练习
# 
# 1.  :numref:`fig_inception`中的Inception块与残差块之间的主要区别是什么？在删除了Inception块中的一些路径之后，它们是如何相互关联的？
# 1. 参考ResNet论文 :cite:`He.Zhang.Ren.ea.2016`中的表1，以实现不同的变体。
# 1. 对于更深层次的网络，ResNet引入了“bottleneck”架构来降低模型复杂性。请试着去实现它。
# 1. 在ResNet的后续版本中，作者将“卷积层、批量规范化层和激活层”架构更改为“批量规范化层、激活层和卷积层”架构。请尝试做这个改进。详见 :cite:`He.Zhang.Ren.ea.2016*1`中的图1。
# 1. 为什么即使函数类是嵌套的，我们仍然要限制增加函数的复杂性呢？

# ## 练习1 Inception块（如:numref:`fig_inception`中所示）与残差块之间的主要区别是什么？在删除了Inception块中的一些路径之后，它们是如何相互关联的？

# - **Inception块**:
#   - 结构：由多个不同的卷积层和池化层并行组成，这些层的输出在通道维度上被连接。
#   - 目的：自适应地从多尺度提取信息。
#   - 特点：可同时处理不同尺寸的卷积核，实现多尺度特征提取。
# 
# - **残差块**:
#   - 结构：包含跳跃连接，允许输入直接“跳过”一个或多个层。
#   - 目的：解决深度网络中的梯度消失和梯度爆炸问题。
#   - 特点：支持更深网络的训练，通过跳跃连接实现恒等映射。
# 
# - **相互关联**：
#   - 在从Inception块中删除一些路径后（例如，只保留$1 \times 1$和$3 \times 3$卷积），Inception块开始与残差块具有相似性，尤其是在考虑到残差块可通过$1 \times 1$卷积改变维度和通道数时。
#   - Inception块依然注重于不同尺度的特征提取，而残差块侧重于信息流的建立和维护，尤其在构建更深网络时。

# In[9]:


## 练习2
# 导入必要的库
import torch
from torch import nn
from torch.nn import functional as F

# 定义ResNet的基本块
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# 定义ResNet模型
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# 创建ResNet-18和ResNet-34模型
def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def resnet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

# 实例化模型
model_resnet18 = resnet18()
model_resnet34 = resnet34()


# In[10]:


## 练习3
import torch
from torch import nn

# 定义Bottleneck块
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# 定义ResNet模型
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# 创建ResNet-50模型的例子
def resnet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])

# 实例化模型
model_resnet50 = resnet50()


# In[11]:


## 练习4
import torch
from torch import nn

# 定义改进后的Bottleneck块
class BottleneckModified(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BottleneckModified, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu3(out)
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu1(out)  # Use relu1 to avoid creating a new ReLU layer

        return out

# 其他部分与之前的ResNet实现相同
# ...

# 创建改进的ResNet模型实例
def resnet50_modified():
    return ResNet(BottleneckModified, [3, 4, 6, 3])

# 实例化模型
model_resnet50_modified = resnet50_modified()


# ## 练习5 限制增加函数复杂性的原因
# 在深度学习和机器学习中，限制模型或函数的复杂性是一个关键考虑因素，即使在涉及嵌套函数类的情况下也是如此。原因主要包括以下几点：
# 
# 1. **避免过拟合**：过于复杂的模型可能会在训练数据上表现出色，但却在新的、未见过的数据上表现不佳。这是因为复杂模型可能会“记住”训练数据的特点，包括噪声，而不是学习到泛化到新数据的底层模式。
# 
# 2. **计算效率**：更复杂的模型通常需要更多的计算资源和时间来训练。这在实际应用中可能是一个限制因素，特别是对于需要快速训练和部署的应用场景。
# 
# 3. **模型泛化**：简单的模型往往更容易泛化到新数据。这是奥卡姆剃刀原则的一个例证，在所有能够解释已知数据的假设中，最简单的假设往往是正确的。
# 
# 4. **理解和调试**：简单的模型通常更容易理解和调试。在实际应用中，能够理解模型是如何做出预测的，对于模型的透明度和可靠性至关重要。
# 
# 5. **维数诅咒**：随着模型复杂性的增加，特别是在参数数量上的增加，模型可能会遭受维数诅咒的影响，即在高维空间中数据点变得稀疏，使得模型训练变得困难。
# 
# 因此，即使在涉及嵌套函数类的情况下，也需要仔细考量在增加模型复杂性方面的取舍。通常，通过交叉验证等技术来选择恰当的模型复杂度，以达到最佳的泛化能力。

# In[ ]:




