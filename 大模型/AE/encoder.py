import torch.nn as nn
from downconvlayer import DownConvLayer

'''
编码器模块，由多个下采样卷积层堆叠而成，用于逐步提取输入数据的特征并进行降维。
'''
class Encoder(nn.Module):
    def __init__(self, dim, layer_num=3):
        super(Encoder, self).__init__()
        # 创建layer_num个DownConvLayer实例，并存储在ModuleList中
        self.convs = nn.ModuleList([DownConvLayer(dim) for _ in range(layer_num)])

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x