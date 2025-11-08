import torch.nn as nn
from MulHdAtt import MultiHeadAttention
from FNN import PositionWiseFFN

class EncoderLayer(nn.Module):    
    def __init__(self, n_head, emb_dim, drop_rate):
        super().__init__()
        # 多头注意力机制层
        self.mha = MultiHeadAttention(n_head, emb_dim, drop_rate)
        # 前馈神经网络层
        self.ffn = PositionWiseFFN(emb_dim, drop_rate)

    def forward(self, xz, mask):
        # xz 的形状为(batch, seq_len, emb_dim)
        # 通过多头注意力机制层，得到context的形状也为(batch, seq_len, emb_dim)
        context = self.mha(xz, xz, xz, mask)
        # 通过前馈神经网络层
        output = self.ffn(context)
        return output
    
class Encoder(nn.Module):
    def __init__(self, n_head, emb_dim, drop_rate, n_layer):
        super().__init__()
        # 定义n_layer个EncoderLayer，保存在ModuleList中
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(n_head, emb_dim, drop_rate) for _ in range(n_layer)]
        )

    def forward(self, xz, mask):
        # 一次通过所有的EncoderLayer
        for encoder in self.encoder_layers:
            xz = encoder(xz, mask)
        return xz   # 返回最终的编码结果，其形状为(batch, seq_len, emb_dim)