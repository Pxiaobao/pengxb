import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
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
cols = ['设备S', '设备Z', '静态抗压强度', '弹性模量', '泊松比', '抗拉强度','黏聚力', '内摩擦角', '拉强比', '脆性指数', '回弹均值', '动态强度', '滑动摩擦系数', '声级',
       '波速', '密度均值', '渗透率', '孔隙度', '粒径']

def linear_func(x, m, b):
    return m * x + b


for i in range(len(cols)):
    for j in range(i+1,len(cols)):
        x = rock_character[cols[i]]
        y = rock_character[cols[j]]
        plt.scatter(x, y, color='blue')
        # 添加拟合直线
        # 使用最小二乘法拟合直线参数
        params, _ = curve_fit(linear_func, x, y)
        y_fit = linear_func(x, params[0], params[1])
        plt.plot(x, y_fit, color='red')
        plt.xlabel(cols[i])
        plt.ylabel(cols[j])
        plt.title('Scatter plot of ' + cols[i] + ' vs ' + cols[j])
        plt.savefig('../output散点图/' + cols[i] + 'vs' + cols[j] + '.png', dpi=300, bbox_inches='tight')
        #清楚图像
        plt.clf()


