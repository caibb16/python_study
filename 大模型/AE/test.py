import torch as th
from downconvlayer import DownConvLayer



# 创建一个 DownConvLayer 实例
down_conv = DownConvLayer(dim=1)
# 创建一个随机输入张量，形状为 (batch_size, channels, height, width)
input_tensor = th.randn(1, 3, 32, 32)  
# 将输入张量传递给 DownConvLayer 的 forward 方法
output_tensor = down_conv(input_tensor)
# 输出降维后的特征图尺寸
print(output_tensor.shape)  