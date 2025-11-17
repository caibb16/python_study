import numpy as np
import matplotlib.pyplot as plt

def cubic_spline_type1(x, y):
    """
    第一型(自然边界条件)3次样条插值
    边界条件: S''(x0) = 0, S''(xn) = 0
    
    参数:
        x: 插值节点 (n+1个)
        y: 函数值 (n+1个)
    
    返回:
        M: 二阶导数值 (n+1个)
        coefficients: 每段的系数 [(a, b, c, d), ...]
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    n = len(x) - 1  # 区间个数
    
    # 计算步长
    h = np.diff(x)  # h[i] = x[i+1] - x[i]
    
    # 构造三对角方程组 AM = d
    # M = [M0, M1, ..., Mn]^T, 其中 Mi = S''(xi)
    
    # 构造系数矩阵A (三对角矩阵)
    A = np.zeros((n+1, n+1))
    d = np.zeros(n+1)
    
    # 第一行: 自然边界条件 M0 = 0
    A[0, 0] = 1.0
    d[0] = 0.0
    
    # 中间行: 2(h[i-1] + h[i])Mi-1 + h[i-1]Mi + h[i]Mi+1 = 6f[xi-1,xi,xi+1]
    for i in range(1, n):
        A[i, i-1] = h[i-1]
        A[i, i] = 2 * (h[i-1] + h[i])
        A[i, i+1] = h[i]
        
        # 计算差商
        d[i] = 6 * ((y[i+1] - y[i]) / h[i] - (y[i] - y[i-1]) / h[i-1])
    
    # 最后一行: 自然边界条件 Mn = 0
    A[n, n] = 1.0
    d[n] = 0.0
    
    # 求解三对角方程组
    M = np.linalg.solve(A, d)
    
    # 计算每段的系数
    # Si(x) = ai + bi(x-xi) + ci(x-xi)^2 + di(x-xi)^3
    coefficients = []
    for i in range(n):
        ai = y[i]
        bi = (y[i+1] - y[i]) / h[i] - h[i] * (2*M[i] + M[i+1]) / 6
        ci = M[i] / 2
        di = (M[i+1] - M[i]) / (6 * h[i])
        coefficients.append((ai, bi, ci, di))
    
    return M, coefficients


def cubic_spline_type2(x, y, dy0, dyn):
    """
    第二型(给定端点导数)3次样条插值
    边界条件: S'(x0) = dy0, S'(xn) = dyn
    
    参数:
        x: 插值节点 (n+1个)
        y: 函数值 (n+1个)
        dy0: 左端点导数值
        dyn: 右端点导数值
    
    返回:
        M: 二阶导数值 (n+1个)
        coefficients: 每段的系数 [(a, b, c, d), ...]
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    n = len(x) - 1  # 区间个数
    
    # 计算步长
    h = np.diff(x)  # h[i] = x[i+1] - x[i]
    
    # 构造三对角方程组 AM = d
    A = np.zeros((n+1, n+1))
    d = np.zeros(n+1)
    
    # 第一行: 边界条件 2h0*M0 + h0*M1 = 6*((y1-y0)/h0 - dy0)
    A[0, 0] = 2 * h[0]
    A[0, 1] = h[0]
    d[0] = 6 * ((y[1] - y[0]) / h[0] - dy0)
    
    # 中间行: h[i-1]*M[i-1] + 2(h[i-1] + h[i])*M[i] + h[i]*M[i+1] = 6*f[xi-1,xi,xi+1]
    for i in range(1, n):
        A[i, i-1] = h[i-1]
        A[i, i] = 2 * (h[i-1] + h[i])
        A[i, i+1] = h[i]
        
        # 计算差商
        d[i] = 6 * ((y[i+1] - y[i]) / h[i] - (y[i] - y[i-1]) / h[i-1])
    
    # 最后一行: 边界条件 h[n-1]*M[n-1] + 2*h[n-1]*M[n] = 6*(dyn - (yn-yn-1)/h[n-1])
    A[n, n-1] = h[n-1]
    A[n, n] = 2 * h[n-1]
    d[n] = 6 * (dyn - (y[n] - y[n-1]) / h[n-1])
    
    # 求解三对角方程组
    M = np.linalg.solve(A, d)
    
    # 计算每段的系数
    # Si(x) = ai + bi(x-xi) + ci(x-xi)^2 + di(x-xi)^3
    coefficients = []
    for i in range(n):
        ai = y[i]
        bi = (y[i+1] - y[i]) / h[i] - h[i] * (2*M[i] + M[i+1]) / 6
        ci = M[i] / 2
        di = (M[i+1] - M[i]) / (6 * h[i])
        coefficients.append((ai, bi, ci, di))
    
    return M, coefficients


def evaluate_spline(x_data, coefficients, x_eval):
    """
    计算样条函数在给定点的值
    
    参数:
        x_data: 插值节点
        coefficients: 每段的系数
        x_eval: 待求值点
    
    返回:
        y_eval: 函数值
    """
    x_data = np.array(x_data)
    n = len(x_data) - 1
    
    # 找到x_eval所在的区间
    if x_eval < x_data[0] or x_eval > x_data[n]:
        raise ValueError(f"x={x_eval} 超出插值区间 [{x_data[0]}, {x_data[n]}]")
    
    # 找到所属区间 [xi, xi+1]
    i = 0
    for j in range(n):
        if x_data[j] <= x_eval <= x_data[j+1]:
            i = j
            break
    
    # 计算 Si(x) = ai + bi(x-xi) + ci(x-xi)^2 + di(x-xi)^3
    ai, bi, ci, di = coefficients[i]
    dx = x_eval - x_data[i]
    y_eval = ai + bi*dx + ci*dx**2 + di*dx**3
    
    return y_eval


def evaluate_spline_array(x_data, coefficients, x_eval_array):
    """
    批量计算样条函数值
    """
    return np.array([evaluate_spline(x_data, coefficients, x) for x in x_eval_array])


def format_significant_figures(x, sig_figs=5):
    """格式化为指定有效数字"""
    if isinstance(x, (list, np.ndarray)):
        return np.array([float(f"{val:.{sig_figs-1}e}") if val != 0 else 0.0 for val in x])
    else:
        return float(f"{x:.{sig_figs-1}e}") if x != 0 else 0.0


def main():
    """示例测试"""
    # 数据点
    x1 = np.arange(11)
    y1 = np.array([2.51, 3.30, 4.04, 4.70, 5.22, 5.54, 5.78, 5.40, 5.57, 5.70, 5.80])
    
    # 端点导数条件
    dy0 = 0.8  # S'(0) = 0.8
    dyn = 0.2  # S'(10) = 0.2
    
    print(f"\n插值节点 x: {x1}")
    print(f"函数值 y: {y1}")
    print(f"\n边界条件:")
    print(f"  S'(0) = {dy0}")
    print(f"  S'(10) = {dyn}")
    
    # 计算样条插值
    M1, coef1 = cubic_spline_type2(x1, y1, dy0, dyn)
    
    print(f"\n二阶导数 M (保留5位有效数字):")
    M1_formatted = format_significant_figures(M1, 5)
    for i, m in enumerate(M1_formatted):
        print(f"  M[{i}] = {m:.5g}")
    
    print(f"\n每段样条系数 [ai, bi, ci, di]:")
    for i, (a, b, c, d) in enumerate(coef1):
        a_fmt = format_significant_figures(a, 5)
        b_fmt = format_significant_figures(b, 5)
        c_fmt = format_significant_figures(c, 5)
        d_fmt = format_significant_figures(d, 5)
        print(f"  S{i}(x): a={a_fmt:.5g}, b={b_fmt:.5g}, c={c_fmt:.5g}, d={d_fmt:.5g}")
    
    # 计算并打印 S(i+0.5), i=0,1,...,9
    print(f"\n" + "=" * 70)
    print("插值结果 S(i+0.5), i=0,1,...,9 (保留5位有效数字):")
    print("=" * 70)
    print(f"\n{'x':<10}{'S(x)':<15}")
    print("-" * 25)
    
    for i in range(10):
        x_test = i + 0.5
        y_test = evaluate_spline(x1, coef1, x_test)
        y_test_fmt = format_significant_figures(y_test, 5)
        print(f"{x_test:<10.1f}{y_test_fmt:<15.5g}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()