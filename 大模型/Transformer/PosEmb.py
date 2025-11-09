import torch as th
import torch.nn as nn
import numpy as np
'''
位置编码层，用于将序列中每个位置的词编码为一个固定大小的向量，该向量包含了词的信息及其在序列中的位置信息。
'''
class PositionEmbedding(nn.Module):
    def __init__(self, max_len, emb_dim, n_vocab):
        super().__init__()
        # 生成位置编码矩阵
        pos = np.expand_dims(np.arange(max_len), axis=1) # [max_len, 1]
        # 使用正弦和余弦函数生成位置编码
        pe = pos / np.power(1000, 2*np.expand_dims(np.arange(emb_dim)//2, axis=0) / emb_dim)
        pe[:, 0::2] = np.sin(pe[:, 0::2])  # 偶数维度使用sin
        pe[:, 1::2] = np.cos(pe[:, 1::2])  # 奇数维度使用cos
        pe = np.expand_dims(pe, 0)  # [1, max_len, emb_dim]
        self.pe = th.from_numpy(pe).type(th.float32)

        # 定义词嵌入层
        self.embeddings = nn.Embedding(n_vocab, emb_dim)
        # 初始化词嵌入层的权重
        self.embeddings.weight.data.normal_(mean=0, std=0.1)

    def forward(self, x):
        # 确保位置编码在与词嵌入权重相同的设备上
        device = self.embeddings.weight.device
        self.pe = self.pe.to(device)
        # 计算输入的词嵌入权重，并加上位置编码
        x_embed = self.embeddings(x) + self.pe   # [batch_size, seq_len, emb_dim]
        return x_embed