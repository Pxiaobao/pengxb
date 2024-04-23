import os
import pandas as pd
import pdb
import pywt
#读取模型数据.xlsx
model_data = pd.read_excel('/Users/pengxb/Documents/project/python_ai/project/矿物分析/data/模型数据副本.xlsx', sheet_name = '设备wob')
import numpy as np
import matplotlib.pyplot as plt

# 假设我们有一段时序数据
data = model_data['材料7']  # 生成一个长度为1000的随机噪声信号作为示例

# # 对数据进行预处理，比如去除直流分量（如果需要），并填充至合适的长度（确保可被2整除以获取准确的频谱）
data -= np.mean(data)
data = np.append(data, np.zeros(len(data) // 2))  # 如果数据长度不是2的幂，则补零

# 执行快速傅里叶变换
sample_rate = 3.0  # 假设采样率为100Hz
fft_data = np.fft.rfft(data)  # 使用rfft获取实数信号的一半频谱
freqs = np.fft.rfftfreq(len(data), d=1/sample_rate)  # 计算相应的频率

# 计算功率谱密度（PSD），这里使用的是幅度平方除以点数（未归一化）
psd = np.abs(fft_data)**2 / len(data)
#归一化
psd = psd / np.sum(psd)


# 绘制频谱
plt.figure(figsize=(10, 6))
plt.plot(freqs, psd)  # 绘制功率谱密度
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density')
plt.title('Single-Sided Power Spectrum of Time Series Data')
plt.grid(True)
plt.show()
