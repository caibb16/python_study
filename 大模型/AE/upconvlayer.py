import torch.nn as nn
import torch.nn.functional as F

'''
上采样卷积层，用于进行特征上采样和提取更细致的特征。
'''
class UpConvLayer(nn.Module):
    def __init__(self, dim):
        super(UpConvLayer, self).__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1) # 卷积层，保持输入输出通道数相同
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')  # 上采样层，放大特征图尺寸为原来的两倍

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.upsample(x)
        return x