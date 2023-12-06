#!/usr/bin/env python
# coding: utf-8

# # 实战Kaggle比赛：预测房价

# In[20]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import xgboost as xgb


# In[21]:


# 加载数据
train_data = pd.read_csv('kaggle_house_pred_train.csv')
test_data = pd.read_csv('kaggle_house_pred_test.csv')

# 填补缺失值
train_data.fillna(train_data.mean(), inplace=True)
test_data.fillna(test_data.mean(), inplace=True)

# 分离特征和目标变量
X = train_data.drop('SalePrice', axis=1)
y = train_data['SalePrice']


# In[22]:


# 选择数值型和类别型特征
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# 创建预处理流程
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


# In[23]:


# 定义模型（这里使用随机森林和XGBoost作为示例）
rf_model = RandomForestRegressor(n_estimators=100)
xgb_model = xgb.XGBRegressor(objective ='reg:squarederror', n_estimators=100)

# 创建管道
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', rf_model)])

xgb_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', xgb_model)])

# 训练模型
rf_pipeline.fit(X, y)
xgb_pipeline.fit(X, y)


# In[24]:


# 使用交叉验证评估模型
rf_scores = cross_val_score(rf_pipeline, X, y, scoring='neg_mean_squared_error')
xgb_scores = cross_val_score(xgb_pipeline, X, y, scoring='neg_mean_squared_error')

# 输出评分
print('Random Forest Mean Squared Error:', np.mean(np.sqrt(-rf_scores)))
print('XGBoost Mean Squared Error:', np.mean(np.sqrt(-xgb_scores)))


# In[25]:


# 使用表现最好的模型对测试集进行预测
predictions = xgb_pipeline.predict(test_data)

# 创建提交文件
output = pd.DataFrame({'Id': test_data.Id, 'SalePrice': predictions})
output.to_csv('submission.csv', index=False)

