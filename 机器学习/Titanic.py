import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report


df = pd.read_csv('Titanic_data/train.csv', index_col='PassengerId')

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

# 使用XGBClassifier和GridSearchCV进行超参数调优
param_grid = {
    'max_depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2],
    'reg_lambda': [0.5, 1.0]
}

xgb_clf = xgb.XGBClassifier(
    objective='binary:logistic',
    nthread=16,
    seed=5,
    eval_metric='logloss'
)

grid_search = GridSearchCV(
    estimator=xgb_clf,
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print("最佳参数:", grid_search.best_params_)
print("最佳得分:", grid_search.best_score_)

# 用最佳参数模型预测验证集
y_pred_label = grid_search.predict(X_valid)
print(classification_report(y_valid, y_pred_label))

# 保存模型
joblib.dump(grid_search, 'best_grid_search.pkl')

# 加载模型
# grid_search = joblib.load('best_grid_search.pkl')