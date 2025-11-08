import torch as th
from Transformer import Transformer
from Transformer import pad_zero
from DateDataset import DateDataset

def evaluate(model, x, y):
    model.eval()
    x = th.from_numpy(pad_zero([x], max_len=MAX_LENGTH)).long().to(device)
    y = th.from_numpy(pad_zero([y], max_len=MAX_LENGTH)).long().to(device)
    decoder_outputs = model(x, y)
    _, topi = decoder_outputs.topk(1)
    decoded_ids = topi.squeeze()
    decoded_words = []
    for idx in decoded_ids:
        decoded_words.append(dataset.index2word[idx.item()])
    return ''.join(decoded_words)

dataset = DateDataset(1000)
device = th.device("cuda" if th.cuda.is_available() else "cpu")
MAX_LENGTH = 11
# 初始化transformer模型
model = Transformer(n_vocab=dataset.num_words, max_len=MAX_LENGTH, n_layer=3, emb_dim=32, n_head=8, drop_rate=0.1, padding_idx=2)
model.to(device)
# 加载模型参数
model.load_state_dict(th.load('transformer.pth', map_location=device))

for i in range(5):  # 循环5个batch进行评估
    predict = evaluate(model, dataset[i][0], dataset[i][1])
    print(f"input:{dataset.date_cn[i]}, target: {dataset.date_en[i]}, predict: {predict}")