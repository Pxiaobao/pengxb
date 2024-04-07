import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2
from matplotlib.patches import Ellipse
rock_character = pd.read_excel('../data/模型数据副本.xlsx',sheet_name = '材料')
# 设置图像大小
plt.figure(figsize=(10, 8))
plt.rcParams['font.sans-serif'] = ['SimHei', 'Songti SC', 'STFangsong']
plt.rcParams['axes.unicode_minus'] = False
cols = ['设备S', '设备Z', '静态抗压强度', '弹性模量', '泊松比', '抗拉强度','黏聚力', '内摩擦角', '拉强比\n', '脆性指数\n', '回弹均值 RM', '动态强度', ' 滑动摩擦系数', '声级',
       '波速', '密度均值', '渗透率', '孔隙度%', '粒径']

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


        # 提取数据列
        data = rock_character[[cols[i], cols[j]]].values
        # 计算样本协方差矩阵
        covariance_matrix = np.cov(data.T)
        # 计算95%置信椭圆的参数
        mean = np.mean(data, axis=0)
        n_samples = len(data)
        n_dim = 2
        confidence = 0.95
        chi2_val = chi2.isf((1 - confidence), n_dim)  # 查找卡方分布临界值
        ellipse_radius = np.sqrt(chi2_val * covariance_matrix.diagonal())
        # 绘制散点图
        plt.scatter(data[:, 0], data[:, 1], alpha=0.5, label='Data Points')
        # 绘制95%置信椭圆
        ellipse = Ellipse(xy=mean, width=2 * ellipse_radius[0], height=2 * ellipse_radius[1],
                        angle=np.rad2deg(np.arccos(covariance_matrix[0, 1] / np.sqrt(covariance_matrix[0, 0] * covariance_matrix[1, 1]))),
                        facecolor='none', edgecolor='r', lw=2)
        plt.gca().add_patch(ellipse)


        plt.xlabel(cols[i])
        plt.ylabel(cols[j])
        plt.title('Scatter plot of ' + cols[i] + ' vs ' + cols[j])
        plt.savefig('../output散点图_ty/Scatter plot of ' + cols[i] + ' vs ' + cols[j] + '.png', dpi=300, bbox_inches='tight')
        #清楚图像
        plt.clf()


        