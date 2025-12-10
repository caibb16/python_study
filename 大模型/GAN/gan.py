import torch
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import os
from torchvision.utils import save_image

# 创建输出目录
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('samples', exist_ok=True)

# 数据加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
data_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=False, transform=transform),
    batch_size=64, shuffle=True  # 若本地无MNIST数据请将 download 设置为 True
)

# 生成器模型
class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(True),
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(True),
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(True),
            torch.nn.Linear(1024, 28*28),
            torch.nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)
    
# 判别器模型
class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(28*28, 1024),
            torch.nn.ReLU(True),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(True),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(True),
            torch.nn.Linear(256, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
    
# 初始化模型
generator = Generator()
discriminator = Discriminator()
# 损失函数和优化器
criterion = torch.nn.BCELoss()
optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0002)
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

# 固定的噪声向量用于生成一致的样本以观察训练进度
fixed_z = torch.randn(64, 128)

# 训练循环
for epoch in range(50):
    for i, (imgs, _) in enumerate(data_loader):
        batch_size = imgs.size(0)
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        # 训练判别器，使其输入真实图像时输出1，输入生成图像时输出0
        outputs = discriminator(imgs.view(batch_size, -1))
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs

        z = torch.randn(batch_size, 128)
        fake_images = generator(z)
        outputs = discriminator(fake_images.detach())
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs

        d_loss = d_loss_real + d_loss_fake
        optimizer_d.zero_grad()
        d_loss.backward()
        optimizer_d.step()

        # 训练生成器，使其生成的图像能骗过判别器
        z = torch.randn(batch_size, 128)
        fake_images = generator(z)
        outputs = discriminator(fake_images)
        g_loss = criterion(outputs, real_labels)

        optimizer_g.zero_grad()
        g_loss.backward()
        optimizer_g.step()

    print(f'Epoch [{epoch+1}/50], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}, '
          f'D(x): {real_score.mean().item():.2f}, D(G(z)): {fake_score.mean().item():.2f}')
    
    # 每5个epoch保存一次检查点和生成样本
    if (epoch + 1) % 5 == 0:
        # 保存模型检查点
        torch.save({
            'epoch': epoch + 1,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_g_state_dict': optimizer_g.state_dict(),
            'optimizer_d_state_dict': optimizer_d.state_dict(),
            'g_loss': g_loss.item(),
            'd_loss': d_loss.item(),
        }, f'checkpoints/checkpoint_epoch_{epoch+1}.pth')
        
        # 生成并保存样本图像
        generator.eval()
        with torch.no_grad():
            fake_images = generator(fixed_z)
            # 将图像从[-1,1]反归一化到[0,1]
            fake_images = fake_images * 0.5 + 0.5
            # 保存图像网格
            save_image(fake_images.view(-1, 1, 28, 28), 
                      f'samples/epoch_{epoch+1}.png', 
                      nrow=8, normalize=False)
        generator.train()
        print(f'已保存检查点和样本图像到 checkpoints/ 和 samples/ 目录')
    
print("训练完成！")

# 保存最终模型
torch.save({
    'epoch': 50,
    'generator_state_dict': generator.state_dict(),
    'discriminator_state_dict': discriminator.state_dict(),
    'optimizer_g_state_dict': optimizer_g.state_dict(),
    'optimizer_d_state_dict': optimizer_d.state_dict(),
}, 'checkpoints/final_model.pth')

# 生成并显示最终样本
generator.eval()
with torch.no_grad():
    fake_images = generator(fixed_z)
    fake_images = fake_images * 0.5 + 0.5  # 反归一化
    save_image(fake_images.view(-1, 1, 28, 28), 'samples/final_samples.png', nrow=8, normalize=False)
    
    # 显示其中一张图像
    plt.figure(figsize=(10, 10))
    plt.imshow(fake_images[0].numpy().reshape(28, 28), cmap='gray')
    plt.title('Generated Sample')
    plt.axis('off')
    plt.savefig('samples/sample_display.png')
    plt.show()

print("模型和样本已保存！")