import seaborn as sns # seaborn: 基于 Matplotlib 的 Python 数据可视化库
import pandas as pd   # Pandas: 基于 Python 的数据分析库，可与Matplotlib、Seaborn等数据可视化库相结合
from sklearn.preprocessing import LabelEncoder, OneHotEncoder   # sklearn: 流行的Python机器学习库


# OneHotEncoder：数据预处理技术，将分类变量转换为独热编码
# le = OneHotEncoder(sparse=False)

# 使用Pandas库中的read_csv()函数读取数据，数据在当前目录下
data = pd.read_csv('./03-ML/00-correlation/03-structure+molecule+chemical-H2O.csv')
data = data.drop(['Order'], axis=1)
# data['structure'] = le.fit_transform(data['structure'])
# data['metal'] = le.fit_transform(data['metal'])
# values = data['structure'].values
# values = values.reshape(len(values), 1)
# data['structure'] = le.fit_transform(values)

# 计算data数据中列与列间的相关性，返回相关系数矩阵
corr = data.corr()

# Matplotlib: 用于创建各种类型的图表和可视化的Python库; pyplot模块提供了类似MATLAB的绘图接口
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 图片显示中文(宋体字体)
plt.rcParams['axes.unicode_minus'] = False  # 减号用ASCII符号来表示，使其与当前环境兼容。

# # =====读取数据和数据预处理=============
# shuju = pd.read_csv('数据.csv')
# print(shuju)
# shuju.isnull().sum()  # 看下有没有缺失值：
# print(shuju)
# shuju.describe()  # 查看数据描述

# =======绘制热图===============
plt.figure(figsize=(25, 40))    # 创建新的图表，尺寸为(25,20)英寸
ax = sns.heatmap(corr, annot=True, annot_kws={"size": 8})  # 将相关系数矩阵可视化为一个颜色编码的矩阵，annot=True表示在热力图上显示数值标签
plt.xticks(fontsize=10, rotation=45) # 设置刻度字体大小
plt.yticks(fontsize=10, rotation=0)
cbar = ax.collections[0].colorbar # 访问由sns.heatmap()函数创建的热力图中的colorbar对象
cbar.ax.tick_params(labelsize=10) # 设置刻度标签的字体大小为20
plt.show() # 图片显示
