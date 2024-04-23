import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

# 创建一个示例数据矩阵
data = np.random.uniform(low=-1, high=1, size=(10, 10))

# 绘制热力图，色带会自动添加
sns.heatmap(data, cmap='coolwarm')

plt.savefig('热力图coolwarm.svg',format='svg')
# 显示图表
plt.show()