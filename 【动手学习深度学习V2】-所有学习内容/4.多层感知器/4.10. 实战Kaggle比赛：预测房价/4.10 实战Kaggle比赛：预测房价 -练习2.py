#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb

# 步骤1：加载数据
train_data = pd.read_csv('kaggle_house_pred_train.csv')
test_data = pd.read_csv('kaggle_house_pred_test.csv')

# 步骤2：数据预处理
# 分离特征和目标变量
X_train = train_data.drop('SalePrice', axis=1)
y_train = train_data['SalePrice']
X_test = test_data.copy() # 可能不需要删除任何列，因为test_data通常不含目标变量

# 填充缺失值
X_train.fillna(X_train.mean(), inplace=True)
X_test.fillna(X_test.mean(), inplace=True)

# 特征选择
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

# 创建列转换器来转换特征
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 步骤3：特征工程（根据需要添加你自己的特征工程步骤）

# 步骤4：模型优化
# 定义一个XGBoost回归模型
xgb_model = xgb.XGBRegressor()

# 设置模型的参数
parameters = {
    'model__objective':['reg:squarederror'],
    'model__learning_rate': [.03, 0.05, .07],
    'model__max_depth': [5, 6, 7],
    'model__min_child_weight': [4],
    'model__subsample': [0.7],
    'model__colsample_bytree': [0.7],
    'model__n_estimators': [500]
}

# 创建一个管道，包含预处理和模型
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', xgb_model)])

# 使用GridSearchCV找到最佳参数
xgb_grid = GridSearchCV(pipeline,
                        parameters,
                        cv = 2,
                        n_jobs = 5,
                        verbose=True)

# 步骤5：训练模型
xgb_grid.fit(X_train, y_train)

# 输出最佳得分和参数
print(xgb_grid.best_score_)
print(xgb_grid.best_params_)

# 步骤6：预测和提交
predictions = xgb_grid.predict(X_test)

# 创建提交文件
output = pd.DataFrame({'Id': test_data['Id'], 'SalePrice': predictions})
output.to_csv('submission.csv', index=False)


# In[ ]:




