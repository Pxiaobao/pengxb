import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pdb
rock_character = pd.read_excel('../data/模型数据副本.xlsx',sheet_name = '材料')
# 设置图像大小
plt.figure(figsize=(10, 8))
plt.rcParams['font.sans-serif'] = ['SimHei', 'Songti SC', 'STFangsong']
plt.rcParams['axes.unicode_minus'] = False
cols = ['设备S', '设备Z', '静态抗压强度', '弹性模量', '泊松比', '抗拉强度','黏聚力', '内摩擦角', '拉强比\n', '脆性指数\n', '回弹均值 RM', '动态强度', ' 滑动摩擦系数', '声级',
       '波速', '密度均值', '渗透率', '孔隙度%', '粒径']

rock_character['内摩擦角'].astype(float)
for col in cols:
    data = rock_character[col].values

    plt.hist(data,  color='green', edgecolor='black')
    plt.title(col)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    #plt.show()
    plt.savefig('../output直方图/' + col + '.png', dpi=300, bbox_inches='tight')
    plt.clf()



