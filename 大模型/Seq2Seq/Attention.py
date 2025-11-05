import torch as th
import torch.nn as nn
import torch.nn.functional as F
'''
Attention机制的实现，用于在序列到序列模型中动态地关注输入序列的不同部分。
'''
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.Wa = nn.Linear(hidden_dim, hidden_dim)
        self.Ua = nn.Linear(hidden_dim, hidden_dim)
        self.Va = nn.Linear(hidden_dim, 1)

    def forward(self, query, keys):
        # query: (batch_size, hidden_dim)
        # keys: (batch_size, seq_len, hidden_dim)

        # 计算注意力权重
        score = self.Va(th.tanh(self.Wa(query) + self.Ua(keys))) # 计算注意力分数，即query和keys的相关性
        score = score.squeeze(2).unsqueeze(1)  # 维度变换以适应softmax计算
        attention_weights = F.softmax(score, dim=-1)

        # 加权求和
        value = keys  # 在这里，value与keys相同
        context = th.bmm(attention_weights, value)

        return context, attention_weights