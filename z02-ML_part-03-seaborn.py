import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import rdBase, Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from collections import Counter
from mordred import Calculator, descriptors
import shap
from tqdm import tqdm

data = pd.read_csv('./03-ML/05-structure+Fingerprint+chemical/02-CatBoost-Estate.csv')
# 设置全局字体为 Arial
plt.rcParams['font.family'] = 'Arial'
#其他参数说明：xlim和ylim设置X和Y轴范围；hue指定变量用于按类分配颜色；marginal_ticks是否在边缘图显示刻度；truncate控制回归线只画在数据范围内
jointgrid = sns.jointplot(x=data['y_test'], y=data['y_predict'],  # 设置xy轴，显示columns名称
            data=data,  # 设置数据
            color='#00A3E0',  # 设置颜色,天蓝色#00A3E0
            height=6,  # 图形高度(以英寸为单位，默认值为6，自动调整为正方形)
            # s = 50, edgecolor = 'w', linewidth = 1,#设置散点大小、边缘颜色及宽度(只针对scatter)
            # stat_func=sci.pearsonr,
            kind='reg',  # 设置类型：散点图'scatter',带回归线的散点图'reg',残差图'resid',核密度估计图'kde',六边形箱图'hex'，直方图'hist'
            #gridsize=30,  # 对于 hex 类型，可以调整六边形的大小
            space=0.1,  # 设置散点图和布局图的间距(默认为0.2)
            ratio=5,  # 散点图与布局图高度比，整型(默认为5)
            #marginal_kws = dict(bins=15, rug =True) #设置柱状图箱数，是否设置rug
                      )
#
jointgrid.set_axis_labels("Simulated I$_{2}$ uptake (mg/g)", "Predicted I$_{2}$ uptake (mg/g)", fontsize=20, fontweight='bold')
#jointgrid.set_axis_labels("Simulated H$_{2}$O uptake (mg/g)", "Predicted H$_{2}$O uptake (mg/g)", fontsize=20, fontweight='bold')

# 获取子图对象并设置刻度的大小
jointgrid.ax_joint.tick_params(axis='x', labelsize=16, direction='in')  # 设置X轴刻度大小
jointgrid.ax_joint.tick_params(axis='y', labelsize=16, direction='in')  # 设置Y轴刻度大小
# 加粗 X 轴的刻度标签
for label in jointgrid.ax_joint.get_xticklabels():
    label.set_fontweight('bold')
# 加粗 Y 轴的刻度标签
for label in jointgrid.ax_joint.get_yticklabels():
    label.set_fontweight('bold')

# 设置边框加粗（设置左、右、上、下四个边框的线宽）
jointgrid.ax_joint.spines['top'].set_linewidth(1.5)
jointgrid.ax_joint.spines['right'].set_linewidth(1.5)
jointgrid.ax_joint.spines['left'].set_linewidth(1.5)
jointgrid.ax_joint.spines['bottom'].set_linewidth(1.5)



#plt.savefig('./03-ML/03-structure+molecule+chemical/04-CatBoost.png', dpi=300,transparent=True)
plt.show()
# marginal_kws = dict(bins=15, rug =True) #设置柱状图箱数，是否设置rug

