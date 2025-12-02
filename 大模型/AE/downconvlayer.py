import torch.nn as nn
import torch.nn.functional as F

'''
下采样卷积层，用于进行特征提取和降维。
'''
class DownConvLayer(nn.Module):
    def __init__(self, dim):
        super(DownConvLayer, self).__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1) # 卷积层，保持输入输出通道数相同
        self.pool = nn.MaxPool2d(2)  # 最大池化层，降采样因子为2，将特征图尺寸减半

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.pool(x)
        return x