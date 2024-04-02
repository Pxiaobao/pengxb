import numpy as np
import pandas as pd
import os
import pandas as pd
import pdb

#读取模型数据.xlsx
df = pd.read_excel('/Users/pengxb/Documents/project/python_ai/project/矿物分析/data/模型数据副本.xlsx',sheet_name='设备wob')


# 计算统计特征
mean_values = df.mean()  # 均值
variance_values = df.var()  # 方差
max_values = df.max()  # 最大值
min_values = df.min()  # 最小值

from scipy.fft import fft

# 傅里叶变换
fft_values = fft(df)

# 计算幅值
amplitude = np.abs(fft_values)

# 因为FFT输出是对称的，我们只需要一半的频谱
amplitude = amplitude[:len(df)//2]

# 计算频域中的最大幅值
max_amplitude = np.max(amplitude, axis=0)

# 注意：FFT结果依赖于数据的长度和采样率，可能需要根据实际情况调整
