import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

rock_character = pd.read_excel('../data/模型数据副本.xlsx',sheet_name = '材料')
device_wob = pd.read_excel('../data/模型数据副本.xlsx',sheet_name = '设备wob')
device_T = pd.read_excel('../data/模型数据副本.xlsx',sheet_name = '设备T')
rock_character = rock_character.drop(['序号', '岩性名称'], axis=1)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Songti SC', 'STFangsong']
plt.rcParams['axes.unicode_minus'] = False
#plt.clf()
joint_columns = ['静态抗压强度', '弹性模量', '泊松比', '抗拉强度', '黏聚力', '内摩擦角', '拉强比',
                 '脆性指数', '回弹均值', '动态强度', '滑动摩擦系数', '声级', '波速', '密度均值', '渗透率', '孔隙度']
#sns.set_theme(font_scale=2)
sns.pairplot(rock_character[joint_columns],kind='reg',diag_kind='hist')
#
plt.savefig('dis.svg',format='svg')
plt.show()

# corr_matrix = rock_character.corr()
# # 设置图像大小
# plt.figure(figsize=(10, 8))
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Songti SC', 'STFangsong']
# plt.rcParams['axes.unicode_minus'] = False

# # 使用Seaborn绘制热力图
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=.5, annot_kws={'size': 10})  # annot=True会显示数字在单元格内
# #保存图片
# plt.savefig('heatmap.svg',format='svg',dpi=300)