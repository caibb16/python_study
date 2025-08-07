import statistics
import random
from knn import x,y

def get_loss(X_, y_, a_, b_):
    """损失函数
    :param X_: 回归模型的自变量
    :param y_: 回归模型的因变量
    :param a_: 回归模型的斜率
    :param b_: 回归模型的截距
    :return: MSE（均方误差）
    """
    y_hat = [a_ * x + b_ for x in X_]
    return statistics.mean([(v1 - v2) ** 2 for v1, v2 in zip(y_, y_hat)])

# 先将最小损失定义为一个很大的值
min_loss, a, b = 1e12, 0, 0

for _ in range(100000):
    # 通过产生随机数的方式获得斜率和截距
    _a, _b = random.random(), random.random() * 4000 - 2000
    # 带入损失函数计算回归模型的MSE
    curr_loss = get_loss(x, y, _a, _b)
    if curr_loss < min_loss:
        # 找到更小的MSE就记为最小损失
        min_loss = curr_loss
        # 记录下当前最小损失对应的a和b
        a, b = _a, _b

print(f'MSE = {min_loss}')
print(f'{a = }, {b = }')