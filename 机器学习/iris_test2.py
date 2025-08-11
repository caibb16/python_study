from sklearn.neighbors import KNeighborsClassifier
from iris_test import X_train,y_train,X_test,y_test


#使用 scikit-learn 来实现 kNN 算法



# 创建模型
model = KNeighborsClassifier()
# 训练模型
model.fit(X_train, y_train)
# 预测结果
y_pred = model.predict(X_test)
print(y_pred == y_test)
print(model.score(X_test, y_test))