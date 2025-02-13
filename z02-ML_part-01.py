# MOF的机器学习相关内容
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
from catboost import CatBoostRegressor
import lightgbm as lgb
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np   # numpy: 科学计算和数据处理库，许多其他科学计算库(SciPy、Pandas)的基础
import json   # json: 轻量级的数据交换格式，被广泛用于数据的传输和存储
import csv    # csv: 提供了一组用于读取和写入CSV文件的工具和方法
from sklearn.ensemble import VotingRegressor         # VotingRegressor: 集成学习方法，用于回归问题
from sklearn.model_selection import GridSearchCV     # GridSearchCV: 系统地搜索模型的最佳参数组合
from sklearn.model_selection import KFold, cross_val_score   # KFold、cross_val_score: 交叉验证分割和交叉验证评估
from sklearn.metrics import mean_absolute_percentage_error   # 回归模型评估指标，用于评估回归模型在预测时的平均绝对百分比误差
import xgboost
import shap   # 解释机器学习模型预测的方法



#data = pd.read_csv('./MOF_ML/03-dataset-8.csv')
data = pd.read_csv('./03-ML/01-structure.csv')

X = data.drop(['Order', 'I2', 'N2', 'O2', 'H2O','Selectivity'], axis=1)  # 删除相应的列，并将结果赋值给X
#X = data.drop(['Order', 'pure_I2'], axis=1)    # 删除相应的列，并将结果赋值给X
y = data['I2']                            # 读取名为"I2_Adsorption"的列，并将结果赋值给y


# 五折交叉验证
# 注意：以下模型中，线性回归模型和SVR模型不属于树模型，故对数值大小比较敏感
# 1使用随机森林模型，使用100棵树（参数范围一般100-1000）
# model = RandomForestRegressor(n_estimators=1000, random_state=42)  # 这里使用100棵树
# 2初始化线性回归模型
# model = LinearRegression()
# 3创建 SVR 模型： 核函数为径向基函数(RBF)；C为惩罚参数(0.01-1000000)；epsilon为不敏感范围大小（0-1），以控制损失函数
# model = SVR(kernel='rbf', C=1000000, epsilon=1)
# 4创建 CatBoost 模型： iterations为迭代次数(100-5000)，learning_rate为学习率(0.01-0.3)，depth为每棵树的最大深度(1-10)，损失函数使用均方根误差
model = CatBoostRegressor(iterations=2000, learning_rate=0.01, depth=12, loss_function='MAE')
# 5创建LightGBM模型: n_estimators为决策树数目(100-5000)，learning_rate为学习率(0.01-0.3)，max_depth为每棵树的最大深度(1-15)
# model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.15, max_depth=5)
# 6使用XGboost，n_estimators为树的数量(100-5000),learning_rate为学习率(0.01-0.3),max_depth为每棵树的最大深度(1-15)
# model = XGBRegressor(n_estimators=5000, learning_rate=0.1, max_depth=7)

# 划分训练集和测试集，测试集数据占20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=16)
# 在训练集上训练模型
model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test)
y_pred = np.abs(y_pred)     # 将y_pred的值取绝对值

# 计算模型性能指标
mae = mean_absolute_error(y_test, y_pred)     # 计算平均绝对误差
mse = mean_squared_error(y_test, y_pred)      # 计算均方误差
r2 = r2_score(y_test, y_pred)                 # 计算决定系数

print(f"Mean absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# xgboost.plot_importance(model)
# plt.show()


# 绘制实际值 vs 预测值图
plt.scatter(y_test, y_pred)       # 使用matplotlib库中scatter函数绘制散点图，展示预测值与实际值的关系
# 使用 numpy.polyfit() 函数拟合回归线
coefficients = np.polyfit(y_test, y_pred, 1)
# 得到拟合的直线方程 y = mx + c
m, c = coefficients
plt.plot(y_test, m * y_test + c, color='red', label='Regression line')
print(f"m= {m}")
print(f"c= {c}")


plt.xlabel("Actual Values")       # X轴标签为"Actual Values"
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.show()
combined_data = np.column_stack((y_test, y_pred))                            #将两列文件合并为一列
#np.savetxt('./03-ML/05-structure+Fingerprint+chemical/02-CatBoost-Estate.csv', combined_data, delimiter=",")
#np.savetxt('./02_RandomForest_test_pred.csv', combined_data, delimiter=",")      # 存储随机森林数据
#np.savetxt('./03_LinearRegression_test_pred.csv', combined_data, delimiter=",")  # 存储线性回归模型数据
#np.savetxt('./04_SVR_test_pred.csv', combined_data, delimiter=",")               # 存储SVR模型数据
#np.savetxt('./05_CatBoost_test_pred.csv', combined_data, delimiter=",")          # 存储CatBoost模型数据
#np.savetxt('./06_LGBMRegressor_test_pred.csv', combined_data, delimiter=",")     # 存储LightGBM模型数据


# 使用SHAP解释器
explainer = shap.TreeExplainer(model)          # 创建 SHAP 解释器
shap_values = explainer.shap_values(X_train)   # 计算 SHAP 值


# 排除不需要的特征
#excluded_features = ['PLD', 'LCD', 'Void', 'Surface', 'Pore_Volume', 'Density', 'I2_Heat', 'H2O_Heat', 'N2_Heat', 'O2_Heat', 'I2_Henry', 'N2_Henry', 'O2_Henry', 'H2O_Henry']
#X_train_filtered = X_train.drop(columns=excluded_features)
# 需要同时从 shap_values 和 X_train_filtered 中删除这些特征
#shap_values_filtered = shap_values[:, X_train.columns.isin(X_train_filtered.columns)]
#shap.summary_plot(shap_values_filtered, X_train_filtered, show=False)        # 绘制汇总图

shap.summary_plot(shap_values, X_train, show=False)        # 绘制汇总图


ax = plt.gca()
# 设置全局字体为 Arial
plt.rcParams['font.family'] = 'Arial'
# 加粗坐标轴
ax.spines['bottom'].set_linewidth(1.5)

# 设置坐标轴字体大小并加粗
ax.tick_params(axis='both', which='major', labelsize=14, width=1, direction='in')

# 设置X轴标题字体大小并加粗
ax.set_xlabel('SHAP Value', fontsize=18, fontweight='bold', labelpad=12)
# 设置X轴坐标的字体大小并加粗
ax.set_xticklabels(ax.get_xticklabels(), fontsize=14, fontweight='bold')
# 设置y轴坐标的字体大小并加粗
ax.set_yticklabels(ax.get_yticklabels(), fontsize=14, fontweight='bold')

# 调整布局
plt.tight_layout()
plt.show()




#np.savetxt('./y_pred_'+id+'.csv', y_pred, delimiter=",")
#np.savetxt('./y_test_' + id + '.csv', y_test, delimiter=",")
#feature_importances = model.feature_importances_
#print("Feature Importances:", feature_importances)
#xgboost.plot_importance(model, max_num_features=10)
#plt.show()


# 计算特征重要性
#shap_values_abs = np.abs(shap_values)
#feature_importance = pd.DataFrame(shap_values_abs, columns=X_test.columns)
# 取平均 SHAP 值并排序
#mean_shap_values = feature_importance.mean(axis=0).sort_values(ascending=False)
# 创建 DataFrame
#shap_df = pd.DataFrame(mean_shap_values).reset_index()
#shap_df.columns = ['Feature', 'Mean SHAP Value']
# 输出到 Excel
#shap_df.to_excel('shap_values.xlsx', index=False)


#shap.summary_plot(shap_values, X_test)