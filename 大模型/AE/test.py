import torch as th
from downconvlayer import DownConvLayer
from upconvlayer import UpConvLayer


# 创建一个 DownConvLayer 实例
down_conv = DownConvLayer(dim=3)
# 创建一个随机输入张量，形状为 (batch_size, channels, height, width)
input_tensor = th.randn(1, 3, 32, 32)  
# 将输入张量传递给 DownConvLayer 的 forward 方法
output_tensor = down_conv(input_tensor)
# 输出降维后的特征图尺寸
print(output_tensor.shape)  


# 创建一个 UpConvLayer 实例
up_conv = UpConvLayer(dim=3)
# 将降维后的特征图传递给 UpConvLayer 的 forward 方法
up_output_tensor = up_conv(output_tensor)
# 输出上采样后的特征图尺寸
print(up_output_tensor.shape)  