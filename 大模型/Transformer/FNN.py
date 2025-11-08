import torch as th
import torch.nn as nn

class PositionWiseFFN(nn.Module):
    def __init__(self, model_dim, dropout=0.0):
        super().__init__()
        # FFN的隐藏层维度，设为模型维度的4倍
        ffn_dim = model_dim * 4
        # 第一个线性层
        self.linear1 = nn.Linear(model_dim, ffn_dim)
        # 第二个线性层
        self.linear2 = nn.Linear(ffn_dim, model_dim)
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        # 层标准化
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        output = self.linear1(x)
        output = th.relu(output)
        output = self.linear2(output)
        output = self.dropout(output)
        output = self.layer_norm(x + output)
        return output     # 返回结果，其形状为(batch, seq_len, model_dim)