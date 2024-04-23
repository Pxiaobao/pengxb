import numpy as np
import pandas as pd
from sklearn.decomposition import PCA  # 注意：此处使用PCA作为KLT的近似实现
import matplotlib.pyplot as plt
# 假设df是一个包含时序数据的DataFrame
# df['ts_column'] 是我们要进行KLT的列
df = pd.read_excel('/Users/pengxb/Documents/project/python_ai/project/矿物分析/data/模型数据副本.xlsx', sheet_name = '设备wob')

del df['时间/s']
# print(df)
# 假设df是一个包含多列时序数据的DataFrame

#只取前100行
df = df.head(500)
#对每一行分别归一化
#df = (df - df.mean()) / df.std()
timeseries_matrix = df.values  # 转换为Numpy数组

# PCA分析
pca = PCA()
pca.fit(timeseries_matrix)

# 提取前k个主成分，作为主要变化趋势
k = 2  # 可根据实际需要选择提取的主成分数量
transformed_data = pca.transform(timeseries_matrix)[:,:k]

# 可视化前两个主成分
pc1_time_series = transformed_data[:, 0]
plt.figure(figsize=(10, 6))
plt.plot(pc1_time_series)
plt.xlabel('Time')
plt.ylabel('First Principal Component Value')
plt.title('Variation Along the First Principal Component')
plt.grid(True)
plt.show()

# 主要趋势可通过第一主成分观察
main_trend = transformed_data[:, 0]