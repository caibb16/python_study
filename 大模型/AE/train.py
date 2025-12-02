import torch
import torchvision
from torchvision import transforms
from aemodel import AutoEncoderModel

# 加载和预处理MNIST数据集
transform = transforms.Compose([transforms.Resize((32, 32)),transforms.ToTensor()])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

# 创建自编码器模型实例、优化器和损失函数
model = AutoEncoderModel(dim=1, layer_num=1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
criterion = torch.nn.MSELoss()
# 创建学习率调度器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
# 训练自编码器模型
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    running_loss = 0.0
    for data in train_loader:
        images, _ = data
        images = images.to(device)


        # 前向传播
        optimizer.zero_grad()
        reconstructed_images = model(images)
        loss = criterion(images, reconstructed_images)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    scheduler.step()
    epoch_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

print('Training Finished')