#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import torch
from d2l import torch as d2l

img = d2l.plt.imread('C:/Users/14591/newfish/【动手学习深度学习】/13. 计算机视觉/catdog.jpg')
h, w = img.shape[:2]
h, w


# In[2]:


def display_anchors(fmap_w, fmap_h, s):
    d2l.set_figsize()
    # 前两个维度上的值不影响输出
    fmap = torch.zeros((1, 10, fmap_h, fmap_w))
    anchors = d2l.multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = torch.tensor((w, h, w, h))
    d2l.show_bboxes(d2l.plt.imshow(img).axes,
                    anchors[0] * bbox_scale)


# In[3]:


display_anchors(fmap_w=4, fmap_h=4, s=[0.15])


# In[4]:


display_anchors(fmap_w=2, fmap_h=2, s=[0.4])


# In[5]:


display_anchors(fmap_w=1, fmap_h=1, s=[0.8])


# 
