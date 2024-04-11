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
# data -= np.mean(data)
# data = np.append(data, np.zeros(len(data) // 2))  # 如果数据长度不是2的幂，则补零

# # 执行快速傅里叶变换
# sample_rate = 100.0  # 假设采样率为100Hz
# fft_data = np.fft.rfft(data)  # 使用rfft获取实数信号的一半频谱
# freqs = np.fft.rfftfreq(len(data), d=1/sample_rate)  # 计算相应的频率

# # 计算功率谱密度（PSD），这里使用的是幅度平方除以点数（未归一化）
# psd = np.abs(fft_data)**2 / len(data)

# # 绘制频谱
# plt.figure(figsize=(10, 6))
# plt.plot(freqs, psd)  # 绘制功率谱密度
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Power Spectral Density')
# plt.title('Single-Sided Power Spectrum of Time Series Data')
# plt.grid(True)
# plt.show()
# 选择一个小波基函数，比如Morlet小波
wavelet_name = 'morl'

# 进行连续小波变换（Continuous Wavelet Transform, CWT）
# 在实际应用中，您需要选择合适的尺度范围
scales = np.arange(1, 1342)  # 可根据实际情况调整尺度范围
cwt_results = pywt.cwt(data, scales, wavelet=wavelet_name)

# 提取尺度轴（对应频率信息）
freqs = pywt.scale2frequency(wavelet_name, scales)

# 绘制时频图
plt.figure(figsize=(10, 6))

# 确保 cwt_results 是一个数组而非元组
if isinstance(cwt_results, tuple):
    # 假设数据是元组的第一个元素
    cwt_results = np.array(cwt_results[0])
else:
    cwt_results = np.array(cwt_results)
# cwt_results 是一个二维数组，每一列代表一个尺度下的小波系数
for i in range(cwt_results.shape[1]):
    plt.plot(data, cwt_results[:, i], label=f'Scale {scales[i]:.2f} Hz')

plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Continuous Wavelet Transform Coefficients Over Time')
plt.legend(title='Scale (Frequency)')
plt.xlim(0, len(data))
plt.ylim(-np.max(np.abs(cwt_results)), np.max(np.abs(cwt_results)))
plt.show()

# 若要绘制类似时频分布的热力图（CWT scalogram）
plt.figure(figsize=(10, 4))
plt.imshow(np.abs(cwt_results), cmap='PRGn', aspect='auto', origin='lower',
           extent=[0, len(data), min(freqs), max(freqs)],
           interpolation='none')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.colorbar(label='Amplitude')
plt.title('Continuous Wavelet Transform Scalogram')
plt.show()