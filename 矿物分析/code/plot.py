import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 假设df是你的DataFrame，包含时间序列数据
# 创建一个示例DataFrame，包含100个数据点
np.random.seed(0)  # 为了可重现性
#读取模型数据.xlsx
df = pd.read_excel('/Users/pengxb/Documents/project/python_ai/project/矿物分析/data/模型数据副本.xlsx',sheet_name='设备wob')
df= df[df['材料2']]

# 设置时间为索引
df = df.set_index()

# 绘制时域幅值变化图
plt.figure(figsize=(10, 6))  # 图片大小
plt.plot(df.index, df['材料A'], label='材料A', color='blue')  # 绘制线图
plt.title('材料A的时域幅值变化')  # 图片标题
plt.xlabel('时间')  # x轴标签
plt.ylabel('幅值')  # y轴标签
plt.legend()  # 显示图例
plt.grid(True)  # 显示网格线

# 格式化x轴日期显示
plt.gcf().autofmt_xdate()

plt.show()
