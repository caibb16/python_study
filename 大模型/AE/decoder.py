import torch.nn as nn
from upconvlayer import UpConvLayer

'''
解码器模块，由多个上采样卷积层堆叠而成，用于逐步恢复输入数据的空间尺寸并提取更细致的特征。
'''
class Decoder(nn.Module):
    def __init__(self, dim, layer_num=3):
        super(Decoder, self).__init__()
        # 创建layer_num个UpConvLayer实例，并存储在ModuleList中
        self.convs = nn.ModuleList([UpConvLayer(dim) for _ in range(layer_num)])
        # 最终卷积层，保持输入输出通道数相同
        self.final_conv = nn.Conv2d(dim, 1, kernel_size=3, padding=1)  
        
    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        x = self.final_conv(x)
        return x