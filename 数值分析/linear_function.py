import numpy as np
import time

class LinearSystemSolver:
    """线性方程组通用求解器"""
    
    def __init__(self, A, b):
        """
        初始化求解器
        
        参数:
            A: 系数矩阵 (n×n)
            b: 右端向量 (n×1)
        """
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.n = len(b)
        
    def gauss_elimination(self):
        """
        Gauss消去法(不选主元)
        
        返回:
            x: 解向量
            steps: 计算步骤信息
        """
        A = self.A.copy()
        b = self.b.copy()
        n = self.n
        
        steps = []
        
        # 消元过程
        for k in range(n - 1):
            # 检查主元
            if abs(A[k, k]) < 1e-10:
                raise ValueError(f"第{k+1}个主元为零,无法使用Gauss消去法")
            
            for i in range(k + 1, n):
                factor = A[i, k] / A[k, k]
                A[i, k:] = A[i, k:] - factor * A[k, k:]
                b[i] = b[i] - factor * b[k]
                steps.append(f"第{k+1}步: 第{i+1}行减去第{k+1}行的{factor:.4f}倍")
        
        # 回代
        x = np.zeros(n)
        x[n-1] = b[n-1] / A[n-1, n-1]
        for i in range(n - 2, -1, -1):
            x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
        
        return x, steps
    
    def gauss_elimination_partial_pivoting(self):
        """
        列主元Gauss消去法
        
        返回:
            x: 解向量
            pivot_history: 主元选择历史
        """
        A = self.A.copy()
        b = self.b.copy()
        n = self.n
        
        pivot_history = []
        
        # 消元过程
        for k in range(n - 1):
            # 列主元选择
            max_row = k
            max_val = abs(A[k, k])
            
            for i in range(k + 1, n):
                if abs(A[i, k]) > max_val:
                    max_val = abs(A[i, k])
                    max_row = i
            
            if abs(A[max_row, k]) < 1e-10:
                raise ValueError(f"矩阵在第{k+1}列为奇异矩阵")
            
            # 交换行
            if max_row != k:
                A[[k, max_row]] = A[[max_row, k]]
                b[[k, max_row]] = b[[max_row, k]]
                pivot_history.append(f"交换第{k+1}行和第{max_row+1}行")
            
            # 消元
            for i in range(k + 1, n):
                factor = A[i, k] / A[k, k]
                A[i, k:] = A[i, k:] - factor * A[k, k:]
                b[i] = b[i] - factor * b[k]
        
        # 回代
        x = np.zeros(n)
        x[n-1] = b[n-1] / A[n-1, n-1]
        for i in range(n - 2, -1, -1):
            x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
        
        return x, pivot_history
    
    def lu_decomposition(self):
        """
        LU分解法(Doolittle分解)
        
        返回:
            x: 解向量
            L: 下三角矩阵
            U: 上三角矩阵
        """
        A = self.A.copy()
        b = self.b.copy()
        n = self.n
        
        L = np.zeros((n, n))
        U = np.zeros((n, n))
        
        # Doolittle分解: A = LU, L对角线为1
        for i in range(n):
            L[i, i] = 1.0
            
            # 计算U的第i行
            for j in range(i, n):
                U[i, j] = A[i, j] - np.dot(L[i, :i], U[:i, j])
            
            # 计算L的第i列
            for j in range(i + 1, n):
                L[j, i] = (A[j, i] - np.dot(L[j, :i], U[:i, i])) / U[i, i]
        
        # 前代: 解Ly = b
        y = np.zeros(n)
        for i in range(n):
            y[i] = b[i] - np.dot(L[i, :i], y[:i])
        
        # 回代: 解Ux = y
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
        
        return x, L, U
    
    def jacobi_iteration(self, x0=None, tol=1e-6, max_iter=1000):
        """
        Jacobi迭代法
        
        参数:
            x0: 初始向量
            tol: 收敛容差
            max_iter: 最大迭代次数
        
        返回:
            x: 解向量
            iter_count: 迭代次数
            residuals: 残差历史
        """
        A = self.A
        b = self.b
        n = self.n
        
        if x0 is None:
            x = np.zeros(n)
        else:
            x = np.array(x0, dtype=float)
        
        x_new = np.zeros(n)
        residuals = []
        
        # 检查对角占优性
        for i in range(n):
            if abs(A[i, i]) < 1e-10:
                raise ValueError(f"第{i+1}行对角元素为零")
        
        for k in range(max_iter):
            for i in range(n):
                sigma = sum(A[i, j] * x[j] for j in range(n) if j != i)
                x_new[i] = (b[i] - sigma) / A[i, i]
            
            residual = np.linalg.norm(x_new - x, ord=np.inf)
            residuals.append(residual)
            
            if residual < tol:
                return x_new, k + 1, residuals
            
            x = x_new.copy()
        
        raise ValueError(f"Jacobi迭代在{max_iter}次后未收敛")
    
    def gauss_seidel_iteration(self, x0=None, tol=1e-6, max_iter=1000):
        """
        Gauss-Seidel迭代法
        
        参数:
            x0: 初始向量
            tol: 收敛容差
            max_iter: 最大迭代次数
        
        返回:
            x: 解向量
            iter_count: 迭代次数
            residuals: 残差历史
        """
        A = self.A
        b = self.b
        n = self.n
        
        if x0 is None:
            x = np.zeros(n)
        else:
            x = np.array(x0, dtype=float)
        
        x_new = x.copy()
        residuals = []
        
        for k in range(max_iter):
            for i in range(n):
                sigma = 0.0
                for j in range(n):
                    if j != i:
                        if j < i:
                            sigma += A[i, j] * x_new[j]
                        else:
                            sigma += A[i, j] * x[j]
                x_new[i] = (b[i] - sigma) / A[i, i]
            
            residual = np.linalg.norm(x_new - x, ord=np.inf)
            residuals.append(residual)
            
            if residual < tol:
                return x_new, k + 1, residuals
            
            x = x_new.copy()
        
        raise ValueError(f"Gauss-Seidel迭代在{max_iter}次后未收敛")
    
    def sor_iteration(self, omega, x0=None, tol=1e-6, max_iter=1000):
        """
        SOR(逐次超松弛)迭代法
        
        参数:
            omega: 松弛因子 (0 < omega < 2)
            x0: 初始向量
            tol: 收敛容差
            max_iter: 最大迭代次数
        
        返回:
            x: 解向量
            iter_count: 迭代次数
            residuals: 残差历史
        """
        A = self.A
        b = self.b
        n = self.n
        
        if x0 is None:
            x = np.zeros(n)
        else:
            x = np.array(x0, dtype=float)
        
        x_new = x.copy()
        residuals = []
        
        for k in range(max_iter):
            for i in range(n):
                sigma = 0.0
                for j in range(n):
                    if j != i:
                        if j < i:
                            sigma += A[i, j] * x_new[j]
                        else:
                            sigma += A[i, j] * x[j]
                
                x_new[i] = (1 - omega) * x[i] + omega * (b[i] - sigma) / A[i, i]
            
            residual = np.linalg.norm(x_new - x, ord=np.inf)
            residuals.append(residual)
            
            if residual < tol:
                return x_new, k + 1, residuals
            
            x = x_new.copy()
        
        raise ValueError(f"SOR迭代在{max_iter}次后未收敛")
    
    def numpy_solve(self):
        """使用NumPy内置求解器"""
        return np.linalg.solve(self.A, self.b)
    
    def verify_solution(self, x):
        """
        验证解的正确性
        
        参数:
            x: 解向量
        
        返回:
            residual: 残差 ||Ax - b||
            relative_error: 相对误差
        """
        Ax = np.dot(self.A, x)
        residual = np.linalg.norm(Ax - self.b)
        relative_error = residual / np.linalg.norm(self.b)
        return residual, relative_error


def format_significant_figures(x, sig_figs=5):
    """格式化为指定有效数字"""
    if isinstance(x, (list, np.ndarray)):
        return np.array([float(f"{val:.{sig_figs-1}e}") if val != 0 else 0.0 for val in x])
    else:
        return float(f"{x:.{sig_figs-1}e}") if x != 0 else 0.0


def main():
    """综合测试示例"""
    print("=" * 90)
    print("线性方程组通用求解器")
    print("=" * 90)
    
    
    # 输入数据
    A1 = [[4, 7],
          [7, 21]]
    b1 = [-0.211, -3.607]
    
    print("系数矩阵A:")
    print(np.array(A1))
    print("\n右端向量b:")
    print(np.array(b1))
    
    solver1 = LinearSystemSolver(A1, b1)
    
    # 方法1: 列主元Gauss消去法
    print("\n方法1: 列主元Gauss消去法")
    start_time = time.time()
    x_gauss, pivot_history = solver1.gauss_elimination_partial_pivoting()
    time_gauss = time.time() - start_time
    x_gauss_fmt = format_significant_figures(x_gauss, 5)
    print(f"解向量 x: {x_gauss_fmt}")
    print(f"计算时间: {time_gauss:.6f} 秒")
    residual, rel_error = solver1.verify_solution(x_gauss)
    print(f"残差: {residual:.2e}, 相对误差: {rel_error:.2e}")
    
    # 方法2: LU分解法
    print("\n方法2: LU分解法")
    start_time = time.time()
    x_lu, L, U = solver1.lu_decomposition()
    time_lu = time.time() - start_time
    x_lu_fmt = format_significant_figures(x_lu, 5)
    print(f"解向量 x: {x_lu_fmt}")
    print(f"计算时间: {time_lu:.6f} 秒")
    residual, rel_error = solver1.verify_solution(x_lu)
    print(f"残差: {residual:.2e}, 相对误差: {rel_error:.2e}")
    
    # 方法3: Gauss-Seidel迭代法
    print("\n方法3: Gauss-Seidel迭代法")
    start_time = time.time()
    x_gs, iter_gs, residuals_gs = solver1.gauss_seidel_iteration(tol=1e-6)
    time_gs = time.time() - start_time
    x_gs_fmt = format_significant_figures(x_gs, 5)
    print(f"解向量 x: {x_gs_fmt}")
    print(f"迭代次数: {iter_gs}, 计算时间: {time_gs:.6f} 秒")
    residual, rel_error = solver1.verify_solution(x_gs)
    print(f"残差: {residual:.2e}, 相对误差: {rel_error:.2e}")
    
    # 方法4: SOR迭代法
    print("\n方法4: SOR迭代法 (omega=1.2)")
    start_time = time.time()
    x_sor, iter_sor, residuals_sor = solver1.sor_iteration(omega=1.2, tol=1e-6)
    time_sor = time.time() - start_time
    x_sor_fmt = format_significant_figures(x_sor, 5)
    print(f"解向量 x: {x_sor_fmt}")
    print(f"迭代次数: {iter_sor}, 计算时间: {time_sor:.6f} 秒")
    residual, rel_error = solver1.verify_solution(x_sor)
    print(f"残差: {residual:.2e}, 相对误差: {rel_error:.2e}")
    



if __name__ == "__main__":
    main()