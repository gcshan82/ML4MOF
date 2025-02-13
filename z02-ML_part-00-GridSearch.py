import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np

# 数据加载
data = pd.read_csv('./03-ML/041-structure+MACCS+chemical.csv')
X = data.drop(['Order', 'I2', 'N2', 'O2', 'H2O', 'Selectivity'], axis=1)
y = data['I2']

# 划分训练集和测试集，测试集占20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=61)

# 设置交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=61)

# 定义参数网格
param_grid = {
    'iterations': [100, 500, 1000, 2000, 5000, 10000],
    'learning_rate': [0.01, 0.02, 0.04, 0.06, 0.08, 1],
    'depth': [6, 7, 8, 9, 10, 11, 12]
}

# 逐个计算每个参数组合，并使用K折交叉验证
for iterations in param_grid['iterations']:
    for learning_rate in param_grid['learning_rate']:
        for depth in param_grid['depth']:
            model = CatBoostRegressor(iterations=iterations,
                                      learning_rate=learning_rate,
                                      depth=depth,
                                      loss_function='MAE',
                                      random_state=61,
                                      verbose=0)

            fold_scores = []
            for train_idx, val_idx in kf.split(X_train):
                # 根据训练集划分训练和验证数据
                X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

                # 训练模型
                model.fit(X_train_fold, y_train_fold)

                # 预测并计算当前折的MAE
                y_pred = model.predict(X_val_fold)
                fold_scores.append(mean_absolute_error(y_val_fold, y_pred))

            # 输出每个超参数组合的性能
            mean_score = np.mean(fold_scores)  # 计算K折交叉验证的平均得分
            print(
                f"Params: iterations={iterations}, learning_rate={learning_rate}, depth={depth} - Mean Test Score: {mean_score:.4f}")