import torch as th
import torch.nn as nn
import numpy as np
class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, model_dim, drop_rate):
       super().__init__()
       # 每个注意力头的维度
       self.head_dim = model_dim // n_head
       # 注意力头的数量
       self.n_head = n_head
       # 模型的维度
       self.model_dim = model_dim
       # 初始化线性变换层，用于生成query, key, value
       self.wq = nn.Linear(model_dim, n_head * self.head_dim)
       self.wk = nn.Linear(model_dim, n_head * self.head_dim)
       self.wv = nn.Linear(model_dim, n_head * self.head_dim)
       # 输出的全连接层
       self.output_dense = nn.Linear(model_dim, model_dim)
       # Dropout层，用于防止过拟合
       self.output_drop = nn.Dropout(drop_rate)
       # 层标准化，用于稳定训练过程
       self.layer_norm = nn.LayerNorm(model_dim)
       self.attention = None

    def forward(self, q, k, v, mask):
        # 保存原始输入q，用于残差连接
        residual = q
        # 分别对输入的q, k, v进行线性变换
        query = self.wq(q)
        key = self.wk(k)
        value = self.wv(v)
        # 对生成的query, key, value进行头分割，以便进行多头注意力计算
        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)
        # 计算上下文向量
        context = self.scaled_dot_product_attention(query, key, value, mask)
        # 对上下文向量进行线性变换
        output = self.output_dense(context)
        # 添加dropout
        output = self.output_drop(output)
        # 添加残差连接并进行层标准化
        output = self.layer_norm(residual + output)
        return output
    
    def split_heads(self, x):
        # 将输入x的形状变为(batch, seq_len, n_head, head_dim)，然后重排为(batch, n_head, seq_len, head_dim)
        x = th.reshape(x, (x.shape[0], x.shape[1], self.n_head, self.head_dim))
        return x.permute(0, 2, 1, 3)
    
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        # 计算缩放因子,等于k的最后一个维度
        dk = th.tensor(k.shape[-1]).type(th.float)
        # 计算注意力得分
        scores = th.matmul(q, k.permute(0, 1, 3, 2)) / (th.sqrt(dk) + 1e-8)
        if mask is not None:
            # 将mask位置的分数设置为负无穷，使其在softmax中接近于0
            scores = scores.masked_fill(mask, -np.inf)
        # 应用softmax函数计算得到注意力权重
        self.attention = th.softmax(scores, dim=-1)
        # 计算上下文向量
        context = th.matmul(self.attention, v)
        # 重排上下文向量的维度并进行维度合并
        context = context.permute(0, 2, 1, 3)
        context = context.reshape((context.shape[0], context.shape[1], -1))
        return context