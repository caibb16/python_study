import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PosEmb import PositionEmbedding
from Encoder import Encoder
from Decoder import Decoder
PAD_TOKEN = 2

class Transformer(nn.Module):
    def __init__(self, n_vocab, max_len, n_layer=6, emb_dim=512, n_head=8, drop_rate=0.1, padding_idx=2):
        super().__init__()
        # 初始化最大长度、填充索引、词汇表大小
        self.max_len = max_len
        self.padding_idx = padding_idx
        self.dec_v_emb = n_vocab
        # 初始化位置嵌入、编码器、解码器和输出层
        self.embedding = PositionEmbedding(max_len, emb_dim, n_vocab)
        self.encoder = Encoder(n_head, emb_dim, drop_rate, n_layer)
        self.decoder = Decoder(n_head, emb_dim, drop_rate, n_layer)
        self.output = nn.Linear(emb_dim, n_vocab)
        # 初始化优化器
        self.opt = th.optim.Adam(self.parameters(), lr=0.002)

    def forward(self, x, y):
        # 对输入和目标进行嵌入
        x_embed, y_embed = self.embedding(x), self.embedding(y)
        # 创建填充掩码
        pad_mask = self._pad_mask(x)
        encoded_z = self.encoder(x_embed, pad_mask)
        yz_look_ahead_mask = self._look_ahead_mask(y)
        decoded_z = self.decoder(y_embed, encoded_z, yz_look_ahead_mask, pad_mask)
        output = self.output(decoded_z)
        return output
    
    def step(self, x, y):
        # 清空梯度
        self.opt.zero_grad()
        # 计算输出，用编码器输入x和解码器输入y的前n-1个token进行前向传播
        logits = self(x, y[:, :-1])  
        # 计算损失，使用交叉熵损失函数计算logits和去掉第一个token的y之间的损失
        loss = F.cross_entropy(logits.reshape(-1, self.dec_v_emb), y[:, 1:].reshape(-1))  
        loss.backward()
        self.opt.step()
        return loss.cpu().detach().numpy(), logits

    def _pad_bool(self, seqs):
        # 创建掩码
        # 返回一个形状与seqs相同的布尔张量，其中的元素为True表示对应位置是填充标记，否则为False
        return th.eq(seqs, self.padding_idx)

    def _pad_mask(self, seqs):
        # 将填充掩码扩展到合适维度
        len_q = seqs.size(1)
        mask = self._pad_bool(seqs).unsqueeze(1).expand(-1, len_q, -1)  # [batch_size, len_q, seq_len]
        return mask.unsqueeze(1)  # [batch_size, 1, len_q, seq_len]
    
    def _look_ahead_mask(self, seqs):
        # 创建前瞻掩码，防止模型在生成序列时看到未来的信息
        device = next(self.parameters()).device
        _, seq_len = seqs.shape
        # 生成一个上三角矩阵，主对角线以上为 1，其余为 0
        mask = th.triu(th.ones((seq_len, seq_len), dtype=th.long), diagonal=1).to(device)
        # 将填充掩码应用到前瞻掩码上
        mask = th.where(self._pad_bool(seqs)[:, None, None, :], 1, mask[None, None, :, :]).to(device)
        # 将 mask 转换为布尔类型，True 表示该位置被屏蔽
        return mask > 0
    
def pad_zero(seqs, max_len):
    # 实现数据填充
    # 初始化形状为[len(seqs), max_len]的矩阵，填充为PAD_TOKEN，这里的len(seqs)等于batch_size
    padded = np.full((len(seqs), max_len), fill_value=PAD_TOKEN, dtype=np.int32)
    for i, seq in enumerate(seqs):
        # 将seqs中的每个seq序列填入padded对应行，实现右侧填充
        padded[i, :len(seq)] = seq
    return padded