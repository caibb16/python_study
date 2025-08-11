from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from scipy import stats
import numpy as np

#用基础的数据科学库 NumPy 和 SciPy 来实现 kNN 算法

def euclidean_distance(u, v):
    """计算两个n维向量的欧式距离"""
    return np.sqrt(np.sum(np.abs(u - v) ** 2))

def make_label(X_train, y_train, X_one, k):
    """
    根据历史数据中k个最近邻为新数据生成标签
    :param X_train: 训练集中的特征
    :param y_train: 训练集中的标签
    :param X_one: 待预测的样本（新数据）特征
    :param k: 邻居的数量
    :return: 为待预测样本生成的标签（邻居标签的众数）
    """
    # 计算x跟每个训练样本的距离
    distes = [euclidean_distance(X_one, X_i) for X_i in X_train]
    # 通过一次划分找到k个最小距离对应的索引并获取到相应的标签
    labels = y_train[np.argpartition(distes, k - 1)[:k]]
    # 获取标签的众数
    return stats.mode(labels).mode

def predict_by_knn(X_train, y_train, X_new, k=5):
    """
    KNN算法
    :param X_train: 训练集中的特征
    :param y_train: 训练集中的标签
    :param X_new: 待预测的样本构成的数组
    :param k: 邻居的数量（默认值为5）
    :return: 保存预测结果（标签）的数组
    """
    return np.array([make_label(X_train, y_train, X, k) for X in X_new])

# 加载鸢尾花数据集
iris = load_iris()
# 特征（150行4列的二维数组，分别是花萼长、花萼宽、花瓣长、花瓣宽）
X = iris.data
# 标签（150个元素的一维数组，包含0、1、2三个值分别代表三种鸢尾花）
y = iris.target
#划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=3)


if __name__ == "__main__":
    # 测试KNN预测
    y_pred = predict_by_knn(X_train, y_train, X_test, k=5)
    print(y_pred == y_test)
    # 计算预测准确率
    accuracy = np.mean(y_pred == y_test)
    print(f'KNN预测准确率: {accuracy:.2%}')