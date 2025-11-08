import torch as th
from torch.utils.data import DataLoader
from Transformer import Transformer
from DateDataset import DateDataset
from Transformer import pad_zero

MAX_LENGTH = 11

# 创建一个数据集，包含1000个样本
dataset = DateDataset(1000)

# 初始化transformer模型
model = Transformer(n_vocab=dataset.num_words, max_len=MAX_LENGTH, n_layer=3, emb_dim=32, n_head=8, drop_rate=0.1, padding_idx=2)
device = th.device("cuda" if th.cuda.is_available() else "cpu")
model.to(device)

# 创建一个数据记载器
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
# 执行10个训练周期
for epoch in range(10):
    for input_tensor, target_tensor, _ in dataloader:
        input_tensor = th.from_numpy(pad_zero(input_tensor,max_len=MAX_LENGTH)).long().to(device)
        target_tensor = th.from_numpy(pad_zero(target_tensor,max_len=MAX_LENGTH+1)).long().to(device)
        # 使用模型的step方法进行一步训练，并获取损失值
        loss, _ = model.step(input_tensor, target_tensor)
    # 打印每个训练周期后的损失值
    print(f"epoch: {epoch+1}, \tloss: {loss}")

# 保存模型参数
th.save(model.state_dict(), 'transformer.pth')
