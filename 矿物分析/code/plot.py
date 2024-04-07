import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

rock_character = pd.read_excel('../data/模型数据副本.xlsx',sheet_name = '材料')
device_wob = pd.read_excel('../data/模型数据副本.xlsx',sheet_name = '设备wob')
device_T = pd.read_excel('../data/模型数据副本.xlsx',sheet_name = '设备T')

rock_character = rock_character.drop(['序号', '岩性名称'], axis=1)

# 设置图像大小
plt.figure(figsize=(10, 8))
plt.rcParams['font.sans-serif'] = ['SimHei', 'Songti SC', 'STFangsong']
plt.rcParams['axes.unicode_minus'] = False

joint_columns=['设备S', '设备Z', '静态抗压强度σc/MPa', '黏聚力',  '拉强比\n/%',
       '动态强度/MPa', ' 滑动摩擦系数μ', '声级LAeq（dB）', '波速vp\n/m·s', '密度均值\n/g·cm-',
       '渗透率mD\n5MPa', '孔隙度%\n5MPa']


#kind表示联合分布图中非对角线图的类型，可选'reg'与'scatter'、'kde'、'hist'，
#'reg'代表在图片中加入一条拟合直线，
#'scatter'就是不加入这条直线,
#'kde'是等高线的形式，'hist'就是类似于栅格地图的形式；
#diag_kind表示联合分布图中对角线图的类型，可选'hist'与'kde'，'hist'代表直方图，'kde'代表直方图曲线化。
# sns.set_theme(font_scale=1.2)
sns.pairplot(rock_character[joint_columns],kind='reg',diag_kind='hist')
plt.show()