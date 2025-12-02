import torch
import torchvision
from torchvision import transforms
from aemodel import AutoEncoderModel
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 加载和预处理MNIST数据集
transform = transforms.Compose([transforms.Resize((32, 32)),transforms.ToTensor()])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

# 创建自编码器模型实例、优化器和损失函数
model = AutoEncoderModel(dim=1, layer_num=1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # 降低学习率
criterion = torch.nn.MSELoss()
# 创建学习率调度器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
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

# 展示原始图片和重建图片
def show_images(model, data_loader, device, num_images=8):
    model.eval()
    with torch.no_grad():
        # 获取一批图片
        images, _ = next(iter(data_loader))
        images = images[:num_images].to(device)
        
        # 重建图片
        reconstructed = model(images)
        
        # 将图片移回CPU并转换为numpy
        images = images.cpu().numpy()
        reconstructed = reconstructed.cpu().numpy()
        
        # 创建对比图
        fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 4))
        
        for i in range(num_images):
            # 原始图片
            axes[0, i].imshow(images[i].squeeze(), cmap='gray')
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('原始图片', fontsize=12)
            
            # 重建图片
            axes[1, i].imshow(reconstructed[i].squeeze(), cmap='gray')
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('重建图片', fontsize=12)
        
        plt.suptitle('自编码器重建效果对比', fontsize=14)
        plt.tight_layout()
        plt.savefig('reconstruction_result.png', dpi=150, bbox_inches='tight')
        plt.show()

# 调用展示函数
show_images(model, train_loader, device)