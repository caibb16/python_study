import torch.optim as optim
import torch.nn as nn
import torch as th
from torch.utils.data import DataLoader
from EncoderRNN import EncoderRNN
from AttentionDecoderRNN import AttentionDecoderRNN
from DateDataset import DateDataset


dataset = DateDataset(1000)

n_epochs = 100
batch_size = 32
MAX_LENGTH = 11
hidden_size = 128
learning_rate = 0.001
criterion = nn.NLLLoss()
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
encoder = EncoderRNN(dataset.num_words, hidden_size)
decoder = AttentionDecoderRNN(hidden_size, dataset.num_words)
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

for i in range(n_epochs):
    total_loss = 0
    for input_tensor, target_tensor, target_length in dataloader:
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)
        loss = criterion(decoder_outputs.view(-1, decoder_outputs.size(-1)), target_tensor.view(-1).long())
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        total_loss += loss.item()

    total_loss /= len(dataloader)
    if i % 10 == 0:
        print(f"Epoch {i}, Loss: {total_loss}")

# 保存模型参数
th.save(encoder.state_dict(), 'encoder.pth')
th.save(decoder.state_dict(), 'decoder.pth')

print("模型已保存。")
