import numpy as np

##使用牛顿迭代法求解非线性方程 f(x) = 0 的数值解

def f(x):
    return np.exp(2*x) - 1- 2*x - 2*x*x

def f_prime(x):
    return np.exp(2*x) * 2 - 2 - 4*x

x = 0.5  # 初始值
tolerance = 1e-3  # 容差，用于控制精度
max_iter = 10  # 最大迭代次数
i = 0

print(f"初始值 x = {x:.5g}", flush=True)

while i < max_iter:
    try:
        denominator = f_prime(x)
        if abs(denominator) < 1e-10:
            print(f"警告：迭代第{i+1}步时分母接近零", flush=True)
            break
            
        x_new = x - f(x) / denominator
        relative_error = abs((x_new - x) / x_new)  # 计算相对误差
        
        print(f"迭代第{i+1}步：x = {x_new:.5g}", flush=True)  # 显示5位有效数字
        
        #if relative_error < tolerance:  # 当相对误差小于容差时停止
        if abs(f(x_new)) < tolerance:  # 当函数值小于容差时停止
            print(f"\n已收敛到五位有效数字！")
            print(f"最终结果：x = {x_new:.5g}")
            break
            
        x = x_new
        i += 1
        
    except Exception as e:
        print(f"计算过程中出错：{e}", flush=True)
        break

if i == max_iter:
    print("\n警告：达到最大迭代次数，可能没有收敛到指定精度")