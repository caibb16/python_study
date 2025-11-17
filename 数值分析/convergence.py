import numpy as np

##分析牛顿法的局部收敛性，确定初始值的收敛区间

def f(x):
    return x*x*x/3 - x

def f_prime(x):
    return x*x - 1

def f_double_prime(x):
    return 4*x


root = 0 #手动指定根位置

# 计算M = max|f''(x)/(2f'(x))|在根附近的值

# 在根附近取一段区间
x_vals = np.linspace(root-1, root+1, 500)
M_vals = np.abs(f_double_prime(x_vals)/(2*f_prime(x_vals)))
M = np.max(M_vals)
print(f"理论收敛区间半径约为：1/M = {1/M:.5f}")
print(f"建议初始值区间：[{root-1/M:.5f}, {root+1/M:.5f}]\n")

