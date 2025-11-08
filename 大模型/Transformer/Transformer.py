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
        x_embed, y_embed = self.embedding(x), self.embedding(y)
        pad_mask = self._pad_mask(x)
        encoded_z = self.encoder(x_embed, pad_mask)
        yz_look_ahead_mask = self._look_ahead_mask(y)
        decoded_z = self.decoder(y_embed, encoded_z, yz_look_ahead_mask, pad_mask)
        output = self.output(decoded_z)
        return output
    
    def step(self, x, y):
        self.opt.zero_grad()
        logits = self(x, y[:, :-1])  
        loss = F.cross_entropy(logits.reshape(-1, self.dec_v_emb), y[:, 1:].reshape(-1))  
        loss.backward()
        self.opt.step()
        return loss.cpu().detach().numpy(), logits

    def _pad_bool(self, seqs):
        return th.eq(seqs, self.padding_idx)

    def _pad_mask(self, seqs):
        len_q = seqs.size(1)
        mask = self._pad_bool(seqs).unsqueeze(1).expand(-1, len_q, -1)
        return mask.unsqueeze(1)
    
    def _look_ahead_mask(self, seqs):
        device = next(self.parameters()).device
        _, seq_len = seqs.shape
        mask = th.triu(th.ones((seq_len, seq_len), dtype=th.long), diagonal=1).to(device)
        mask = th.where(self._pad_bool(seqs)[:, None, None, :], 1, mask[None, None, :, :]).to(device)
        return mask > 0
    
def pad_zero(seqs, max_len):
    padded = np.full((len(seqs), max_len), fill_value=PAD_TOKEN, dtype=np.int32)
    for i, seq in enumerate(seqs):
        padded[i, :len(seq)] = seq
    return padded