from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
'''
聚类是一种无监督学习，因为它不需要预先定义的标签，
只是根据数据特征去学习，通过度量特征相似度或者距离，然后把已知的数据集划分成若干个不同的类别。
K-Means 是一种基于原型的分区聚类方法，其目标是将数据集划分为K个簇，并使每个簇内的数据点尽可能相似。
'''
# 加载数据集
iris = load_iris()
X = iris.data

# 创建KMeans对象
km_cluster = KMeans(
    n_clusters=3,       # k值（簇的数量）
    max_iter=30,        # 最大迭代次数
    n_init=10,          # 初始质心选择尝试次数
    init='k-means++',   # 初始质心选择算法
    algorithm='elkan',  # 是否使用三角不等式优化
    tol=1e-4,           # 质心变化容忍度
    random_state=3      # 随机数种子
)
# 训练模型
km_cluster.fit(X)
print(km_cluster.labels_)           # 分簇的标签
print(km_cluster.cluster_centers_)  # 各个质心的位置
print(km_cluster.inertia_)          # 样本到质心的距离平方和

colors = ['#FF6969', '#050C9C', '#365E32']
markers = ['o', 'x', '^']

plt.figure(dpi=200)
for i in range(len(km_cluster.cluster_centers_)):
    samples = X[km_cluster.labels_ == i]  #布尔索引
    print(markers[i])
    plt.scatter(samples[:, 2], samples[:, 3], marker=markers[i], color=colors[i])
    plt.scatter(km_cluster.cluster_centers_[i, 2], km_cluster.cluster_centers_[i, 3], marker='*', color='r', s=120)

plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.show()