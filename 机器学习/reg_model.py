import ssl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
'''
回归模型是一种统计分析方法，用于建立自变量与因变量之间的关系。
它通过拟合数据来预测目标变量的值，在经济学、工程、医学等领域有着广泛的应用，
可以帮助决策者进行数据驱动的预测和分析。
'''

#导入汽车 MPG 数据集
ssl._create_default_https_context = ssl._create_unverified_context
df = pd.read_csv('https://archive.ics.uci.edu/static/public/9/data.csv')

# 删除指定的列
df.drop(columns=['car_name'], inplace=True)
# 计算相关系数矩阵
df.corr()
# 删除有缺失值的样本
df.dropna(inplace=True)
# 将origin字段处理为类别类型
df['origin'] = df['origin'].astype('category') 
# 将origin字段处理为独热编码
df = pd.get_dummies(df, columns=['origin'], drop_first=True)
print(df)

# 分割数据集为训练集和测试集
X, y = df.drop(columns='mpg').values, df['mpg'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=3)
#采用线性回归模型（LinearRegression使用最小二乘法计算回归模型的参数）
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# 查看模型参数
print('回归系数:', model.coef_)
print('截距:', model.intercept_)
#评估模型
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred) #模型拟合的效果越好，r2 的值就越接近 1。通常 r2≥0.8 时，我们认为模型的拟合效果已经很不错了

print(f'均方误差: {mse:.4f}')
print(f'平均绝对误差: {mae:.4f}')
print(f'决定系数: {r2:.4f}')