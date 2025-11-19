import numpy as np

'''
用于对return_list进行平滑处理的函数
保证输出长度与输入相同
'''
def moving_average(a, window_size): 
    # 计算累积和,用于滑动窗口
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    # 计算完整窗口的平均值
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    
    # 计算开头和结尾部分的平均值
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    
    return np.concatenate((begin, middle,end))