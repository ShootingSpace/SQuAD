#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''  
统计分析SQUAD并记录参数，
后面可能用于指导模型调参，如模型的输出大小限制值，
绘制上下文段长度，问题长度和答案长度的直方图。
'''
import numpy as np
import matplotlib.pyplot as plt
import os

file_names = ['train.answer', 'train.context','train.question']

for name in file_names:
    data_path = os.path.join(os.getcwd(), 'data','squad', name)
    _ = []
    line_counts = 0
    with open(data_path, 'rb') as f:
        for line in f:
            _.append(len(line.split()))
            line_counts += 1
    
    fig = plt.figure(figsize=(6, 4))
    plt.hist(_, bins=100)  # arguments are passed to np.histogram
    plt.title(name)
    plt.show()
    fig.savefig(name.replace(".", "-")+ '-histogram.pdf')
print('line counts {}'.format(line_counts))

