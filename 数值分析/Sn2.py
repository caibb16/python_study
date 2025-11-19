import numpy as np

def calculate_sn_forward(N):
    """
    从小到大计算Sn=∑(1/(j^2-1)), j从2到N (单精度)
    """
    Sn = np.float32(0.0)
    for j in range(2, N + 1):
        Sn = np.float32(Sn + np.float32(1.0) / np.float32(j * j - 1))
    return float(Sn)


def calculate_sn_backward(N):
    """
    从大到小计算Sn=∑(1/(j^2-1)), j从N到2 (单精度)
    """
    Sn = np.float32(0.0)
    for j in range(N, 1, -1):
        Sn = np.float32(Sn + np.float32(1.0) / np.float32(j * j - 1))
    return float(Sn)


def calculate_sn_analytical(N):
    """
    使用部分分式分解的解析公式 (双精度作为精确值)
    1/(j^2-1) = 1/(j-1)(j+1) = 1/2 * (1/(j-1) - 1/(j+1))
    Sn = 1/2 * [(1/1 - 1/3) + (1/2 - 1/4) + ... + (1/(N-1) - 1/(N+1))]
    """
    # 利用裂项相消 (使用双精度作为精确参考值)
    Sn = 0.5 * (1 + 1/2 - 1/N - 1/(N+1))
    return Sn


def calculate_sn_analytical_single(N):
    """
    使用部分分式分解的解析公式 (单精度)
    """
    Sn = np.float32(0.5) * (np.float32(1.0) + np.float32(0.5) - 
                             np.float32(1.0)/np.float32(N) - 
                             np.float32(1.0)/np.float32(N+1))
    return float(Sn)


def count_significant_figures(computed, exact):
    """
    计算有效位数
    通过比较计算值和精确值的相对误差
    """
    if exact == 0:
        return 0
    
    relative_error = abs((computed - exact) / exact)
    
    if relative_error == 0:
        return 7  # 单精度浮点数最多约7位有效数字
    
    # 有效位数 ≈ -log10(相对误差)
    sig_figs = -np.log10(relative_error)
    return max(0, min(sig_figs, 7))  # 单精度最多7位


def main():
    print("=" * 90)
    print("计算 Sn = ∑(1/(j²-1)), j从2到N (单精度)")
    print("=" * 90)
    
    # 测试不同的N值
    N_values = [10**2, 10**4, 10**6]
    
    print(f"\n{'N':<12}{'从小到大(单精度)':<22}{'从大到小(单精度)':<22}{'解析值(双精度)':<22}{'有效位数':<12}")
    print("-" * 90)
    
    for N in N_values:
        # 三种计算方法
        sn_forward = calculate_sn_forward(N)
        sn_backward = calculate_sn_backward(N)
        sn_analytical = calculate_sn_analytical(N)
        
        # 计算有效位数(以双精度解析值为精确值)
        sig_figs_forward = count_significant_figures(sn_forward, sn_analytical)
        sig_figs_backward = count_significant_figures(sn_backward, sn_analytical)
        
        print(f"{N:<12}{sn_forward:<22.10f}{sn_backward:<22.10f}{sn_analytical:<22.15f}{sig_figs_forward:<12.1f}")
    
    # 详细分析
    print("\n" + "=" * 90)
    print("详细分析")
    print("=" * 90)
    
    for N in N_values:
        print(f"\n当 N = {N:,}:")
        print("-" * 50)
        
        sn_forward = calculate_sn_forward(N)
        sn_backward = calculate_sn_backward(N)
        sn_analytical = calculate_sn_analytical(N)
        sn_analytical_single = calculate_sn_analytical_single(N)
        
        print(f"  从小到大计算(单精度): {sn_forward:.10f}")
        print(f"  从大到小计算(单精度): {sn_backward:.10f}")
        print(f"  解析值(单精度):       {sn_analytical_single:.10f}")
        print(f"  解析值(双精度):       {sn_analytical:.15f}")
        
        # 计算误差
        error_forward = abs(sn_forward - sn_analytical)
        error_backward = abs(sn_backward - sn_analytical)
        
        print(f"\n  绝对误差(从小到大): {error_forward:.2e}")
        print(f"  绝对误差(从大到小): {error_backward:.2e}")
        
        # 相对误差
        rel_error_forward = error_forward / abs(sn_analytical)
        rel_error_backward = error_backward / abs(sn_analytical)
        
        print(f"  相对误差(从小到大): {rel_error_forward:.2e}")
        print(f"  相对误差(从大到小): {rel_error_backward:.2e}")
        
        # 有效位数
        sig_figs_forward = count_significant_figures(sn_forward, sn_analytical)
        sig_figs_backward = count_significant_figures(sn_backward, sn_analytical)
        
        print(f"\n  有效位数(从小到大): {sig_figs_forward:.1f} 位")
        print(f"  有效位数(从大到小): {sig_figs_backward:.1f} 位")
    
    # 单精度与双精度对比
    print("\n" + "=" * 90)
    print("单精度 vs 双精度对比")
    print("=" * 90)
    
    print(f"\n{'N':<15}{'单精度(从小到大)':<25}{'双精度(从小到大)':<25}{'精度损失':<20}")
    print("-" * 85)
    
    for N in N_values:
        sn_single = calculate_sn_forward(N)
        # 计算双精度版本
        sn_double = 0.0
        for j in range(2, N + 1):
            sn_double = sn_double + 1.0 / (j * j - 1)
        
        precision_loss = abs(sn_single - sn_double)
        print(f"{N:<15,}{sn_single:<25.10f}{sn_double:<25.15f}{precision_loss:<20.2e}")
    
    # 公式推导说明
    print("\n" + "=" * 90)
    print("公式推导")
    print("=" * 90)
    print("""
使用部分分式分解:
    1/(j²-1) = 1/((j-1)(j+1)) = 1/2 * [1/(j-1) - 1/(j+1)]

因此:
    Sn = ∑(j=2 to N) 1/(j²-1)
       = 1/2 * ∑(j=2 to N) [1/(j-1) - 1/(j+1)]
       = 1/2 * [(1/1 - 1/3) + (1/2 - 1/4) + (1/3 - 1/5) + ... + (1/(N-1) - 1/(N+1))]

裂项相消后:
    Sn = 1/2 * [1 + 1/2 - 1/N - 1/(N+1)]

当N→∞时, Sn → 3/4 = 0.75

单精度浮点数(float32):
    - 有效数字: 约6-7位十进制数字
    - 机器精度: ε ≈ 1.19×10⁻⁷
    
双精度浮点数(float64):
    - 有效数字: 约15-16位十进制数字
    - 机器精度: ε ≈ 2.22×10⁻¹⁶
    """)
    
    # 收敛分析
    print("=" * 90)
    print("收敛分析 (单精度 vs 双精度)")
    print("=" * 90)
    print(f"\n{'N':<15}{'单精度Sn':<20}{'双精度Sn':<20}{'与3/4的差(单精度)':<25}")
    print("-" * 80)
    
    for N in [10, 100, 1000, 10000, 100000, 1000000]:
        sn_single = calculate_sn_analytical_single(N)
        sn_double = calculate_sn_analytical(N)
        diff_single = abs(sn_single - 0.75)
        print(f"{N:<15,}{sn_single:<20.10f}{sn_double:<20.15f}{diff_single:<25.2e}")


if __name__ == "__main__":
    main()