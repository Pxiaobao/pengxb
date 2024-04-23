import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import math
rock_character = pd.read_excel('../data/模型数据副本.xlsx',sheet_name = '材料')
device_wob = pd.read_excel('../data/模型数据副本.xlsx',sheet_name = '设备wob')
device_T = pd.read_excel('../data/模型数据副本.xlsx',sheet_name = '设备T')
rock_character = rock_character.drop(['序号', '岩性名称'], axis=1)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Songti SC', 'STFangsong']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({
    "font.size": 35,  # 修改全局字体大小
})

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

joint_columns = ['设备S', '设备Z', '静态抗压强度', '弹性模量', '泊松比', '抗拉强度', '黏聚力', '内摩擦角', '回弹均值',
       '动态强度', '滑动摩擦系数', '声级', '波速', '密度均值', '渗透率', '孔隙度', '标定温度']
# 计算相关系数矩阵
correlation_matrix = rock_character[joint_columns].corr()

#布局更紧凑一点，中间加入网格线
g = sns.PairGrid(rock_character[joint_columns],layout_pad=0)
#设置子图之间的间隙
g.figure.subplots_adjust(wspace=0.0, hspace=0.0)
g.map_lower(sns.regplot)  # 下三角
g.map_diag(sns.histplot) # 对角线

for i in range(len(joint_columns)):
    for j in range(i + 1, len(joint_columns)):
        r = correlation_matrix.iloc[i, j]
        col_x, col_y = joint_columns[i], joint_columns[j]
        
        # 获取当前子图的ax对象
        ax = g.axes[i][j]
                #删除原子图
        #根据 r的值设置子图背景色，颜色越大越深
        if r>0:
            ax.set_facecolor((0.0, 0.2, 1.0, 0.5 + 0.5 * np.sin(np.pi * (2 * abs(r) - 1))      ))
        else:
            ax.set_facecolor((0.8, 0.0, 0.0, 0.5 + 0.5 * np.sin(np.pi * (2 * abs(r) - 1))  ))
        #ax.set_facecolor('lightgrey')
        # 
        ax.text(0.7, 0.5, f'  R = {r:.2f}', ha='right', va='top', transform=ax.transAxes, fontsize=24)

  # 上三角




# # 若要在图上标注，可能需要进一步自定义每个子图的内容
# for i in range(len(joint_columns)):
#     for j in range(i + 1, len(joint_columns)):
#        r = correlation_matrix.iloc[i, j]
#        col_x, col_y = joint_columns[i], joint_columns[j]
        
#        # 获取当前子图的ax对象
#        ax = g.axes[i][j]
#        #删除原子图
#        ax.clear()
#        # 这里只是简单地在右上角显示R^2，实际位置和样式可能需要调整
#        ax.text(0.5, 0.5, f'R = {r:.2f}', ha='right', va='top', transform=ax.transAxes, fontsize=16)


plt.savefig('联合分布图20240421.svg',format='svg')
plt.show()


#plt.clf()
""" joint_columns = ['设备S', '设备Z', '静态抗压强度', '弹性模量', '泊松比', '抗拉强度', '黏聚力', '内摩擦角', '回弹均值',
       '动态强度', '滑动摩擦系数', '声级', '波速', '密度均值', '渗透率', '孔隙度', '标定温度']
#joint_columns = ['设备S', '设备Z', '静态抗压强度', '弹性模量', '泊松比']
#sns.set_theme(font_scale=2)
sns.pairplot(rock_character[joint_columns],kind='reg',diag_kind='hist')
#
plt.savefig('联合分布图20240408.svg',format='svg')
plt.show() """
