import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取CSV文件
file_path = './03-ML/00-correlation/05-structure+chemical+MACCS-2.csv'
data = pd.read_csv(file_path)

# 计算整个数据集的相关性矩阵
correlation_matrix = data.corr()

# 提取第Y列与第X列之间的相关性矩阵
correlation_subset = correlation_matrix.iloc[6:20, 20:40]

# Matplotlib: 用于创建各种类型的图表和可视化的Python库; pyplot模块提供了类似MATLAB的绘图接口
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 图片显示中文(宋体字体)
plt.rcParams['axes.unicode_minus'] = False  # 减号用ASCII符号来表示，使其与当前环境兼容。


# =======绘制热图===============
plt.figure(figsize=(25, 40))    # 创建新的图表，尺寸为(25,20)英寸
ax = sns.heatmap(correlation_subset, annot=True, annot_kws={"size": 8})  # 将相关系数矩阵可视化为一个颜色编码的矩阵，annot=True表示在热力图上显示数值标签
plt.xticks(fontsize=10, rotation=45) # 设置刻度字体大小
plt.yticks(fontsize=10, rotation=0)
cbar = ax.collections[0].colorbar # 访问由sns.heatmap()函数创建的热力图中的colorbar对象
cbar.ax.tick_params(labelsize=10) # 设置刻度标签的字体大小为20
plt.show() # 图片显示
