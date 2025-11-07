import torch as th
import torch.nn as nn
import torch.nn.functional as F
from EncoderRNN import EncoderRNN
from Attention import Attention
'''
实现带有注意力机制的解码器RNN
'''
SOS_TOKEN = 0  # Start of Sequence token
EOS_TOKEN = 1  # End of Sequence token
PAD_TOKEN = 2  # Padding token
MAX_LENGTH = 11  # 最大序列长度
class AttentionDecoderRNN(nn.Module):
    def __init__(self, hidden_dim, output_dim, dropout_p=0):
        super(AttentionDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_dim, hidden_dim)
        self.attention = Attention(hidden_dim)
        self.rnn = nn.RNN(2 * hidden_dim, hidden_dim, dropout=dropout_p, batch_first=True) # RNN 输入为嵌入向量和上下文向量的拼接，维度翻倍
        self.out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = th.empty(batch_size, 1, dtype=th.long).fill_(SOS_TOKEN)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []
        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(decoder_input, decoder_hidden, encoder_outputs)
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()

        decoder_outputs = th.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = th.cat(attentions, dim=1)
        return decoder_outputs, decoder_hidden, attentions

    def forward_step(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input))
        query = hidden.permute(1, 0, 2) # 将hidden的第0维和第1维交换，以适应注意力机制
        context, attn_weights = self.attention(query, encoder_outputs) # 接收到的encoder_hidden作为query，encoder_outputs作为key和value
        input_rnn = th.cat((embedded, context), dim=2)
        output, hidden = self.rnn(input_rnn, hidden)
        output = self.out(output)
        return output, hidden, attn_weights

#test
def test_attention_decoder_rnn():
    encoder = EncoderRNN(input_dim=10, hidden_dim=5)
    input_vector = th.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    encoder_outputs, encoder_hidden = encoder(input_vector)

    decoder = AttentionDecoderRNN(hidden_dim=5, output_dim=10)
    target_vector = th.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    decoder_outputs, decoder_hidden, attentions = decoder(encoder_outputs, encoder_hidden, target_vector)

    print("解码器输出向量的维度:", decoder_outputs.shape)  # (batch_size, seq_len, output_dim)
    print("注意力权重的维度:", attentions.shape)  # (batch_size, seq_len, encoder_seq_len)

__all__ = ['AttentionDecoderRNN', 'test_attention_decoder_rnn']