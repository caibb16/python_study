import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# 加载数据并做同样的预处理（确保特征顺序和训练时一致）
test = pd.read_csv('Titanic_data/test.csv', index_col='PassengerId')
# 处理缺失值
test['Age'] = test.Age.fillna(test.Age.median())
test['Fare'] = test.Fare.fillna(test.Fare.median())
test['Embarked'] = test.Embarked.fillna(test.Embarked.mode()[0])
test['Cabin'] = test.Cabin.replace(r'.+', '1', regex=True).replace(np.nan, 0).astype('i8')
# 特征缩放
scaler = StandardScaler()
test[['Fare', 'Age']] = scaler.fit_transform(test[['Fare', 'Age']])
# 处理类别
test = pd.get_dummies(test, columns=['Sex', 'Embarked'], drop_first=True)
# 特征构造
title_mapping = {
    'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Dr': 4, 'Rev': 5, 'Col': 6, 'Major': 7, 
    'Mlle': 8, 'Ms': 9, 'Lady': 10, 'Sir': 11, 'Jonkheer': 12, 'Don': 13, 'Dona': 14, 'Countess': 15
}
test['Title'] = test['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip()).map(title_mapping).fillna(-1)
test['FamilySize'] = test['SibSp'] + test['Parch'] + 1
# 删除多余特征
test.drop(columns=['Name', 'Ticket', 'SibSp', 'Parch'], inplace=True)


# 使用XGBoost模型
passenger_id, X_test = test.index, test

# 加载模型
grid_search = joblib.load('best_grid_search.pkl')
print("最佳参数:", grid_search.best_params_)
print("最佳得分:", grid_search.best_score_)
# 在测试集上评估
y_test_pred = grid_search.predict(X_test)


# 生成提交文件
result = pd.DataFrame({
    'PassengerId': passenger_id,
    'Survived': y_test_pred
})

result.to_csv('submission.csv', index=False)
