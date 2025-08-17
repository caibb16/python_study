import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial  # 导入多项式拟合函数


# 利用收集到的月收入和网购支出的历史数据来建立一个预测模型，以达到通过某人的月收入预测他网购支出金额的目标
# 下面是我们收集到的收入和网购支出的数据，保存在两个数组中
x = np.array([
    25000, 15850, 15500, 20500, 22000, 20010, 26050, 12500, 18500, 27300,
    15000,  8300, 23320,  5250,  5800,  9100,  4800, 16000, 28500, 32000,
    31300, 10800,  6750,  6020, 13300, 30020,  3200, 17300,  8835,  3500
])
y = np.array([
    2599, 1400, 1120, 2560, 1900, 1200, 2320,  800, 1650, 2200,
     980,  580, 1885,  600,  400,  800,  420, 1380, 1980, 3999,
    3800,  725,  520,  420, 1200, 4020,  350, 1500,  560,  500
])

# 使用numpy的多项式拟合函数来拟合数据
data = Polynomial.fit(x, y, deg=1).convert().coef
a = data[1]
b = data[0]
print(f'拟合直线方程: y = {a} * x + {b}')   

# 绘制散点图
plt.scatter(x, y, color='blue')
plt.scatter(x, a * x + b, color='red')
# 绘制拟合直线
plt.plot(x, a * x + b, color='darkcyan')
plt.show()
