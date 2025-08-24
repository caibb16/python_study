import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import classification_report

df = pd.read_csv('Titanic_data/train.csv', index_col='PassengerId')
df.info()

# 处理缺失值
df['Age'] = df.Age.fillna(df.Age.median())
df['Embarked'] = df.Embarked.fillna(df.Embarked.mode()[0])
df['Cabin'] = df.Cabin.replace(r'.+', '1', regex=True).replace(np.nan, 0).astype('i8')
# 对年龄和船票进行标准化
scaler = StandardScaler()
df[['Fare', 'Age']] = scaler.fit_transform(df[['Fare', 'Age']])
# 类别变量独热编码
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
# 提取头衔和家庭规模特征
title_mapping = {
    'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Dr': 4, 'Rev': 5, 'Col': 6, 'Major': 7, 
    'Mlle': 8, 'Ms': 9, 'Lady': 10, 'Sir': 11, 'Jonkheer': 12, 'Don': 13, 'Dona': 14, 'Countess': 15
}
df['Title'] = df['Name'].map(
    lambda x: x.split(',')[1].split('.')[0].strip()
).map(title_mapping).fillna(-1)
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
# 删除不必要的列
df.drop(columns=['Name', 'SibSp', 'Parch', 'Ticket'], inplace=True)

# 划分训练集和验证集
X, y = df.drop(columns='Survived'), df.Survived
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.9, random_state=3)

# 使用XGBoost进行训练和预测
# 将数据处理成数据集格式DMatrix格式
dm_train = xgb.DMatrix(X_train, y_train)
dm_valid = xgb.DMatrix(X_valid)

# 设置模型参数
params = {
    'booster': 'gbtree',             # 用于训练的基学习器类型
    'objective': 'binary:logistic',  # 指定模型的损失函数
    'gamma': 0.1,                    # 控制每次分裂的最小损失函数减少量
    'max_depth': 10,                 # 决策树最大深度
    'lambda': 0.5,                   # L2正则化权重
    'subsample': 0.8,                # 控制每棵树训练时随机选取的样本比例
    'colsample_bytree': 0.8,         # 用于控制每棵树或每个节点的特征选择比例
    'eta': 0.05,                     # 学习率
    'seed': 3,                       # 设置随机数生成器的种子
    'nthread': 16,                   # 指定了训练时并行使用的线程数
}

model = xgb.train(params, dm_train, num_boost_round=200)
y_pred = model.predict(dm_valid)
# 将预测的概率转换为类别标签
y_pred_label = (y_pred > 0.5).astype('i8')
print(classification_report(y_valid, y_pred_label))