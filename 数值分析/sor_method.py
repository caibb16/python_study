import numpy as np

def sor_method(A, b, omega, x0=None, tol=1e-6, max_iter=1000):
    """
    SOR(逐次超松弛迭代)方法求解线性方程组 Ax = b
    
    参数:
        A: 系数矩阵 (n x n)
        b: 右端向量 (n x 1)
        omega: 松弛因子 (0 < omega < 2)
        x0: 初始向量 (默认为零向量)
        tol: 收敛容差
        max_iter: 最大迭代次数
    
    返回:
        x: 解向量
        iter_count: 实际迭代次数
        residuals: 每次迭代的残差
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    
    # 初始化
    if x0 is None:
        x = np.zeros(n)
    else:
        x = np.array(x0, dtype=float)
    
    x_new = x.copy()
    residuals = []
    
    # 检查对角元素是否为零
    for i in range(n):
        if abs(A[i, i]) < 1e-10:
            raise ValueError(f"第{i+1}行对角元素为零,无法使用SOR方法")
    
    # 迭代过程
    for k in range(max_iter):
        for i in range(n):
            # 计算 sigma = sum(a_ij * x_j)
            sigma = 0.0
            for j in range(n):
                if j != i:
                    if j < i:
                        # 使用已更新的x_new
                        sigma += A[i, j] * x_new[j]
                    else:
                        # 使用旧的x
                        sigma += A[i, j] * x[j]
            
            # SOR迭代公式: x_new[i] = (1-omega)*x[i] + omega*(b[i] - sigma)/A[i,i]
            x_new[i] = (1 - omega) * x[i] + omega * (b[i] - sigma) / A[i, i]
        
        # 计算残差 ||x_new - x||
        residual = np.linalg.norm(x_new - x, ord=np.inf)
        residuals.append(residual)
        
        # 检查收敛性
        if residual < tol:
            return x_new, k + 1, residuals
        
        # 更新x
        x = x_new.copy()
    
    return x_new, max_iter, residuals


def format_significant_figures(x, sig_figs=5):
    """
    将数组中的数字格式化为指定位数的有效数字
    
    参数:
        x: 输入数组
        sig_figs: 有效数字位数
    
    返回:
        格式化后的数组
    """
    return np.array([float(f"{val:.{sig_figs-1}e}") if val != 0 else 0.0 for val in x])


def main():
    """示例测试 - 测试不同松弛因子"""
    print("=" * 70)
    print("SOR方法求解线性方程组 - 松弛因子优化测试")
    print("=" * 70)
    
    # 测试方程组
    A = [
        [31, -13, 0, 0, 0, -10, 0, 0, 0],
        [-13, 35, -9, 0, -11, 0, 0, 0, 0],
        [0, -9, 31, -10, 0, 0, 0, 0, 0],
        [0, 0, -10, 79, -30, 0, 0, 0, -9],
        [0, 0, 0, -30, 57, -7, 0, -5, 0],
        [0, 0, 0, 0, -7, 47, -30, 0, 0],
        [0, 0, 0, 0, 0, -30, 41, 0, 0],
        [0, 0, 0, 0, -5, 0, 0, 27, -2],
        [0, 0, 0, -9, 0, 0, 0, -2, 29],
    ]
    b = [-15, 27, -23, 0, -20, 12, -7, 7, 10]
    
    print("\n系数矩阵A:")
    print(np.array(A))
    print("\n右端向量b:")
    print(np.array(b))
    
    # 容许误差
    tolerance = 0.5 * 1e-5
    print(f"\n容许误差: {tolerance:.2e}")
    print(f"\n开始测试松弛因子 omega = i/50, i = 1, 2, ..., 99")
    print("=" * 70)
    
    # 存储结果
    results = []
    
    # 打印表头
    print(f"\n{'松弛因子':<15}{'迭代次数':<15}{'收敛状态':<15}")
    print("-" * 45)
    
    # 测试不同的松弛因子
    for i in range(1, 100):
        omega = i / 50.0
        
        try:
            x, iter_count, residuals = sor_method(A, b, omega=omega, tol=tolerance, max_iter=1000)
            
            # 检查是否真正收敛
            if residuals[-1] < tolerance:
                status = "收敛"
                results.append({
                    'omega': omega,
                    'iter_count': iter_count,
                    'x': x,
                    'residual': residuals[-1]
                })
                print(f"{omega:<15.4f}{iter_count:<15}{status:<15}")
            else:
                status = "未收敛"
                print(f"{omega:<15.4f}{iter_count:<15}{status:<15}")
        
        except Exception as e:
            print(f"{omega:<15.4f}{'N/A':<15}发散")
    
    # 找出最佳松弛因子(迭代次数最少)
    if results:
        print("\n" + "=" * 70)
        print("收敛结果汇总")
        print("=" * 70)
        
        best_result = min(results, key=lambda x: x['iter_count'])
        
        print(f"\n最佳松弛因子: {best_result['omega']:.4f}")
        print(f"迭代次数: {best_result['iter_count']}")
        
        
        # 格式化并打印解向量
        x_formatted = format_significant_figures(best_result['x'], sig_figs=5)
        print(f"\n解向量 x (保留5位有效数字):")
        for i, val in enumerate(x_formatted):
            print(f"  x[{i}] = {val:.5g}")
    else:
        print("\n警告: 所有测试的松弛因子均未能使方程收敛!")


if __name__ == "__main__":
    main()