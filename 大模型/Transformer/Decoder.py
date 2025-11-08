import torch.nn as nn
from MulHdAtt import MultiHeadAttention
from FNN import PositionWiseFFN

class DecoderLayer(nn.Module):
    def __init__(self, n_head, model_dim, drop_rate):
        super().__init__()
        # 定义两个多头注意力机制层
        self.mha = nn.ModuleList([
            MultiHeadAttention(n_head, model_dim, drop_rate) for _ in range(2)
        ])
        # 定义前馈神经网络层
        self.ffn = PositionWiseFFN(model_dim, drop_rate)

    def forward(self, yz, xz, yz_look_ahead_mask, xz_pad_mask):
        # 执行第一个多头注意力机制层计算，处理目标序列的自注意力
        dec_output = self.mha[0](yz, yz, yz, yz_look_ahead_mask)
        # 执行第二个多头注意力机制层计算，处理目标序列与源序列之间的注意力
        dec_output = self.mha[1](dec_output, xz, xz, xz_pad_mask)
        # 通过前馈神经网络层
        output = self.ffn(dec_output)
        return output
    
class Decoder(nn.Module):
    def __init__(self, n_head, model_dim, drop_rate, n_layer):
        super().__init__()
        # 定义n_layer个DecoderLayer，保存在ModuleList中
        self.num_layers = n_layer
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(n_head, model_dim, drop_rate) for _ in range(n_layer)]
        )

    def forward(self, yz, xz, yz_look_ahead_mask, xz_pad_mask):
        # 依次通过所有的DecoderLayer
        for decoder in self.decoder_layers:
            yz = decoder(yz, xz, yz_look_ahead_mask, xz_pad_mask)
        return yz   # 返回最终的解码结果，其形状为(batch, seq_len, model_dim)