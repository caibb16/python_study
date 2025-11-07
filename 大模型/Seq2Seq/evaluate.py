import torch as th
import numpy as np
from EncoderRNN import EncoderRNN
from AttentionDecoderRNN import AttentionDecoderRNN
from DateDataset import DateDataset


dataset = DateDataset(1000)


hidden_size = 128
encoder = EncoderRNN(dataset.num_words, hidden_size)
decoder = AttentionDecoderRNN(hidden_size, dataset.num_words)

# 加载训练好的模型参数
encoder.load_state_dict(th.load('encoder.pth'))
decoder.load_state_dict(th.load('decoder.pth'))

def evaluate(encoder, decoder, x):
    encoder.eval()
    decoder.eval()
    encoder_outputs, encoder_hidden = encoder(th.tensor(np.array([x])))
    start = th.ones(x.shape[0], 1)
    start[:,0] = th.tensor(0).long()
    decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden)
    _, topi = decoder_outputs.topk(1)
    decoded_ids = topi.squeeze()
    decoded_words = []
    for idx in decoded_ids:
        decoded_words.append(dataset.index2word[idx.item()])
    return ''.join(decoded_words)

for i in range(5):
    predict = evaluate(encoder, decoder, dataset[i][0])
    print(f"input:{dataset.date_cn[i]}, target: {dataset.date_en[i]}, predict: {predict}")
