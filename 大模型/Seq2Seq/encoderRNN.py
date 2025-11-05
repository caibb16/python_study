import torch as th
import torch.nn as nn

'''
EncoderRNN类，用于将输入序列编码为隐藏状态表示。
'''

class EncoderRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_p=0):
        super(EncoderRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(input_dim, hidden_dim) # 输入维度到隐藏维度的嵌入层
        self.rnn = nn.RNN(hidden_dim, hidden_dim, dropout=dropout_p, batch_first=True) # RNN层
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        output, hidden = self.rnn(x)
        return output, hidden
    
#test
def test_encoder_rnn():
    encoder = EncoderRNN(input_dim=10, hidden_dim=5)  # 词表大小为10，隐藏层大小为5
    input_vector = th.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    embedded = encoder.embedding(input_vector) # 获取嵌入向量,嵌入层将输入序列的每个元素映射到一个隐藏维度的向量
    output, hidden = encoder(input_vector)
    print("输入序列的维度：", input_vector.shape)
    print("输入序列经过嵌入层后的维度:", embedded.shape)
    print("输出向量的维度:", output.shape)  #output.shape = (batch_size, seq_len, hidden_dim)
    print("隐藏状态的维度:", hidden.shape)  #hidden.shape = (num_layers, batch_size, hidden_dim)
    
__all__ = ['EncoderRNN', 'test_encoder_rnn']