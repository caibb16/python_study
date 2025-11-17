import numpy as np

def gauss_elimination_with_partial_pivoting(A, b):
    """
    列主元Gauss消去法求解线性方程组 Ax = b
    
    参数:
        A: 系数矩阵 (n x n)
        b: 右端向量 (n x 1)
    
    返回:
        x: 解向量
    """
    # 将A和b转换为浮点数矩阵,并复制以避免修改原始数据
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    
    # 构造增广矩阵
    Ab = np.column_stack([A, b])
    
    # 消元过程
    for k in range(n - 1):
        # 列主元选择:在第k列找到绝对值最大的元素
        max_row = k
        max_val = abs(Ab[k, k])
        
        for i in range(k + 1, n):
            if abs(Ab[i, k]) > max_val:
                max_val = abs(Ab[i, k])
                max_row = i
        
        # 检查主元是否为零
        if abs(Ab[max_row, k]) < 1e-10:
            raise ValueError(f"矩阵在第{k+1}列为奇异矩阵,无法求解")
        
        # 交换行
        if max_row != k:
            Ab[[k, max_row]] = Ab[[max_row, k]]
            print(f"交换第{k+1}行和第{max_row+1}行")
        
        # 消元:将第k列下方的元素变为0
        for i in range(k + 1, n):
            factor = Ab[i, k] / Ab[k, k]
            Ab[i, k:] = Ab[i, k:] - factor * Ab[k, k:]
    
    # 检查最后一个主元
    if abs(Ab[n-1, n-1]) < 1e-10:
        raise ValueError("矩阵为奇异矩阵,无法求解")
    
    # 回代过程
    x = np.zeros(n)
    x[n-1] = Ab[n-1, n] / Ab[n-1, n-1]
    
    for i in range(n - 2, -1, -1):
        sum_ax = np.dot(Ab[i, i+1:n], x[i+1:n])
        x[i] = (Ab[i, n] - sum_ax) / Ab[i, i]
    
    return x


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
    
    print("系数矩阵A:")
    print(np.array(A))
    print("\n右端向量b:")
    print(np.array(b))
    
    x = gauss_elimination_with_partial_pivoting(A, b)
    x_formatted = format_significant_figures(x, sig_figs=5)
    print("\n解向量x (保留5位有效数字):")
    print(x_formatted)

    # 验证解
    print("\n验证 Ax:")
    verification1 = np.dot(A, x_formatted)
    print(verification1)
    



if __name__ == "__main__":
    main()