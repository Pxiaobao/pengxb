#cnn加bp
import numpy as np
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

rock_character = pd.read_excel('../data/模型数据副本.xlsx',sheet_name = '材料', engine='openpyxl')
device_wob = pd.read_excel('../data/模型数据副本.xlsx',sheet_name = '设备wob', engine='openpyxl')
device_T = pd.read_excel('../data/模型数据副本.xlsx',sheet_name = '设备T', engine='openpyxl')
device_wob['time'] = [int(temp) for temp in device_wob['时间/s']]
device_T['time'] = [int(temp) for temp in device_T['时间/s']]
device_S = pd.read_excel('../data/模型数据副本.xlsx',sheet_name = '设备S', engine='openpyxl')
device_Z = pd.read_excel('../data/模型数据副本.xlsx',sheet_name = '设备Z', engine='openpyxl')
del device_wob['时间/s']
del device_T['时间/s']
columns = ['材料2','材料4','材料5','材料7','材料8','材料10','材料11','材料13','材料55','材料21','材料22']
res_df = pd.DataFrame(columns=['time1','time2','time3','time4','time5','time6','time7','time8','time9','time10','res'])
T_samples = []
wob_samples = []
outputs = []
for time in range(0,len(device_T)-1,10):
    device_wob_temp = device_wob[time:time+10]
    device_T_temp = device_T[time:time+10]
    for column in columns:
        T_samples.append(device_T_temp[column].values)
        wob_samples.append(device_wob_temp[column].values)
        output = rock_character[rock_character['岩性名称']==column]['静态抗压强度'].values[0]
        outputs.append(output)

# 将T和wob样本转换为多通道输入
T_data = np.array(T_samples).reshape(-1, len(T_samples[0]), 1)  # (样本数, 时间步长, 1)
wob_data = np.array(wob_samples).reshape(-1, len(wob_samples[0]), 1)  # (样本数, 时间步长, 1)

# 将T和wob拼接为一个多通道输入
input_data = np.concatenate((T_data, wob_data), axis=-1)  # 形状变为(样本数, 时间步长, 特征数*2)

# 目标输出整理为二维数组
output_data = np.array(outputs).reshape(-1, 1)



# 构建模型
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_data.shape[1:]))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=1, activation='linear'))  # 对于回归问题，最后的输出层通常使用线性激活函数

model.compile(loss='mean_squared_error', optimizer='adam')  # 使用均方误差损失函数和Adam优化器

#划分训练集和测试集

X_train, X_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.2, random_state=42)
# 训练模型
model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=0)  

# 预测
prediction = model.predict(X_test)
#评估模型精度mse、rmse、r2

mse = mean_squared_error(y_test, prediction)
print("均方误差：", mse)
rmse = np.sqrt(mse)
print("均方根误差：", rmse)

r2 = r2_score(y_test, prediction)
print("R2：", r2)
