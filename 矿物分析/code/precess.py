import os
import pandas as pd
import pdb

#读取模型数据.xlsx
model_data = pd.read_excel('/Users/pengxb/Documents/project/python_ai/project/矿物分析/data/模型数据副本.xlsx')
#去除序号和岩性名称两列
model_data = model_data.drop(['序号', '岩性名称'], axis=1)
corr_matrix = model_data.corr()

import seaborn as sns
import matplotlib.pyplot as plt

# 设置图像大小
plt.figure(figsize=(10, 8))
plt.rcParams['font.sans-serif'] = ['SimHei', 'Songti SC', 'STFangsong']
plt.rcParams['axes.unicode_minus'] = False

# 使用Seaborn绘制热力图
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=.5)  # annot=True会显示数字在单元格内
#plt.show()
plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')
