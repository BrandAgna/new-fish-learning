#!/usr/bin/env python
# coding: utf-8

# ## 练习
# 
# 1. 根据我们在 :numref:`sec_alexnet`中的讨论，深度神经网络学习图像特征级别抽象层次，随网络深度的增加而升级。在多尺度目标检测中，不同尺度的特征映射是否对应于不同的抽象层次？为什么？
# 1. 在 :numref:`subsec_multiscale-anchor-boxes`中的实验里的第一个尺度（`fmap_w=4, fmap_h=4`）下，生成可能重叠的均匀分布的锚框。
# 1. 给定形状为$1 \times c \times h \times w$的特征图变量，其中$c$、$h$和$w$分别是特征图的通道数、高度和宽度。怎样才能将这个变量转换为锚框类别和偏移量？输出的形状是什么？

# 1. **不同尺度的特征映射与抽象层次**:
#    
#    在多尺度目标检测中，不同尺度的特征映射通常对应于不同的抽象层次。在深度神经网络中，随着网络深度的增加，特征映射的空间分辨率通常会降低，而抽象层次会升高。这意味着较低层的特征映射捕捉到更多细节和纹理信息，而较高层的特征映射则包含更高层次的语义信息（如物体部件或整个物体的形状）。
# 
#    在多尺度目标检测中，利用这些不同层次的特征映射可以更好地检测不同大小的物体。例如，较小的物体可能在较低层的高分辨率特征映射中更容易被检测到，而较大的物体可能在较高层的低分辨率特征映射中更容易被检测到。

# In[6]:


## 练习 2
get_ipython().run_line_magic('matplotlib', 'inline')
import torch
from d2l import torch as d2l

# 读取图像
img = d2l.plt.imread('C:/Users/14591/newfish/【动手学习深度学习】/13. 计算机视觉/catdog.jpg')
h, w = img.shape[:2]

# 显示锚框的函数
def display_anchors(fmap_w, fmap_h, sizes):
    d2l.set_figsize()
    # 前两个维度上的值不影响输出
    fmap = torch.zeros((1, 10, fmap_h, fmap_w))
    # 使用多种尺寸和宽高比
    ratios = [1, 2, 0.5]
    anchors = d2l.multibox_prior(fmap, sizes=sizes, ratios=ratios)
    bbox_scale = torch.tensor((w, h, w, h))
    d2l.show_bboxes(d2l.plt.imshow(img).axes,
                    anchors[0] * bbox_scale)

# 在4x4特征图上生成锚框
display_anchors(fmap_w=4, fmap_h=4, sizes=[0.15, 0.2, 0.25])


# 3. **将特征图变量转换为锚框类别和偏移量**:
#    
#    给定形状为$1 \times c \times h \times w$的特征图变量，可以通过将其转换为两个输出来表示锚框类别和偏移量：
#    
#    - **锚框类别**（分类）：将特征图通过一个卷积层转换，输出通道数为锚框数量乘以类别数。例如，如果有$k$个锚框和$n$个类别，则输出形状为$1 \times (k \times n) \times h \times w$。
#    - **锚框偏移量**（回归）：将特征图通过另一个卷积层转换，输出通道数为锚框数量乘以4（每个锚框的偏移量：$\Delta x, \Delta y, \Delta w, \Delta h$）。输出形状为$1 \times (k \times 4) \times h \times w$。
# 
#    这些输出随后可以用于目标检测任务中的分类和边界框回归。

# In[ ]:




