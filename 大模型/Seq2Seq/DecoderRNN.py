import torch as th
import torch.nn as nn
import torch.nn.functional as F
from EncoderRNN import EncoderRNN

'''
DecoderRNN类，用于根据编码器的输出和隐藏状态逐步生成目标序列。
'''
SOS_TOKEN = 0  # Start of Sequence token
EOS_TOKEN = 1  # End of Sequence token
PAD_TOKEN = 2  # Padding token
MAX_LENGTH = 10  # 最大序列长度

class DecoderRNN(nn.Module):
    def __init__(self, output_dim, hidden_dim, dropout_p=0):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_dim, hidden_dim) # 嵌入层
        self.rnn = nn.RNN(hidden_dim, hidden_dim, dropout=dropout_p, batch_first=True) # RNN层
        self.out = nn.Linear(hidden_dim, output_dim) # 线性层，用于将RNN的输出映射回词表大小（nn.Linear作用：最后一维由 hidden_dim 变为 output_dim）

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = th.empty(batch_size, 1, dtype=th.long).fill_(SOS_TOKEN)  # 初始化解码器的第一个输入为SOS_TOKEN标记，作为序列的开始
        decoder_hidden = encoder_hidden  # 初始化解码器隐藏状态为编码器的最后隐藏状态
        decoder_outputs = []
        
        for i in range(MAX_LENGTH):  # 循环生成每个时间步的输出
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:  # 强制学习
                decoder_input = target_tensor[:, i].unsqueeze(1)  # 使用真实目标作为下一个输入
            else:
                _, topi = decoder_output.topk(1)  # 选择概率最高的词作为下一个输入
                decoder_input = topi.detach()  # 分离梯度
        
        decoder_outputs = th.cat(decoder_outputs, dim=1)  # 拼接所有时间步的输出
        '''
        dim=0 会沿着批次维度拼接
        dim=1 沿着序列维度拼接,将同一样本的不同时间步的输出组合成完整序列
        dim=2 会沿着特征维度拼接
        '''
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)  # 应用log softmax
        return decoder_outputs, decoder_hidden, None
    
    def forward_step(self, x, hidden):
        x = self.embedding(x)
        x = F.relu(x)
        x, hidden = self.rnn(x, hidden)
        output = self.out(x)
        return output, hidden
    
#test
if __name__ == "__main__":
    encoder = EncoderRNN(input_dim=10, hidden_dim=5)  # 词表大小为10，隐藏层大小为5
    input_vector = th.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    encoder_outputs, encoder_hidden = encoder(input_vector)

    decoder = DecoderRNN(output_dim=10, hidden_dim=5)  # 词表大小为10，隐藏层大小为5
    target_vector = th.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    decoder_outputs, decoder_hidden, _ = decoder(encoder_outputs, encoder_hidden,target_vector)

    print("解码器输出的形状:", decoder_outputs.shape)  # output.shape = (batch_size, seq_len, output_dim)


