import torch
import torch.nn as nn # 导入torch.nn模块，用于定义神经网络层
import torch.optim as optim # 导入torch.optim模块，用于定义优化器
import torchvision # 导入torchvision模块，用于加载和处理数据集
import torchvision.transforms as transforms # 导入torchvision.transforms模块，用于定义数据增强和预处理

import matplotlib.pyplot as plt
import matplotlib

from torch.utils.data import random_split
5


# 指定一个支持中文的字体（Windows 下通常有 SimHei、Microsoft YaHei）
matplotlib.rcParams['font.sans-serif'] = ['SimHei']        # 用黑体
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False

# 定义Transformer块，这是ViT的核心组件
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout=0.1): # dim是输入的维度，heads是注意力头的数量，mlp_dim是MLP的维度，dropout是Dropout的dropout率，这一段是在初始化TransformerBlock类
        super().__init__() # 继承nn.Module类
        # 第一个层归一化
        self.norm1 = nn.LayerNorm(dim)
        # 多头注意力机制，batch_first=True表示输入张量的第一个维度是batch_size
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        # 第二个层归一化
        self.norm2 = nn.LayerNorm(dim)
        # 多层感知机，用于特征转换
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),  # 第一个全连接层，扩大维度
            nn.GELU(),                # GELU激活函数
            nn.Dropout(dropout),      # Dropout防止过拟合
            nn.Linear(mlp_dim, dim),  # 第二个全连接层，恢复原始维度
            nn.Dropout(dropout)       # Dropout防止过拟合
        )

    def forward(self, x, return_attn=False): #这一段是TransformerBlock类的forward方法，用于前向传播，x代表输入的特征张量
        # 残差连接 + 多头注意力
        x_norm = self.norm1(x) # x_norm是归一化后的特征张量
        attn_output, attn_weights = self.attn(x_norm, x_norm, x_norm)# 多头注意力机制，返回的attn_output是(batch_size, seq_len, dim)，有两个值，第一个是输出，第二个是权重
        x = x + attn_output # 残差连接
        # 残差连接 + MLP
        x = x + self.mlp(self.norm2(x))
        if return_attn:
            return x, attn_weights
        return x

# 定义Vision Transformer模型
class ViT(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_classes=1000, dim=768, depth=12, heads=12, mlp_dim=3072, dropout=0.1): #这一段是ViT类的初始化方法，用于初始化ViT类，image_size是输入的图像尺寸，patch_size是patch的大小，num_classes是分类的数量，dim是嵌入的维度，depth是Transformer块的数量，heads是注意力头的数量，mlp_dim是MLP的维度，dropout是Dropout的dropout率
        super(ViT, self).__init__()
        # 保存基本参数
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes

        # 确保图像尺寸能被patch大小整除
        assert image_size % patch_size == 0, "图像尺寸必须能被patch大小整除"
        # 计算patch数量
        self.num_patches = (image_size // patch_size) ** 2

        # 将图像转换为patch嵌入
        
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),  # 使用卷积进行patch划分
            nn.Flatten(start_dim=2, end_dim=3),  # 展平patch
        )

        # 分类token，用于最终的分类
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        # 可学习的位置编码（符合原ViT）
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        
        # Dropout层，防止过拟合
        self.dropout = nn.Dropout(dropout)
        
        # 堆叠多个Transformer块
        self.transformer = nn.Sequential(*[
            TransformerBlock(dim, heads, mlp_dim, dropout) for _ in range(depth)
        ])

        # 最终的分类头
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),  # 层归一化
            nn.Linear(dim, num_classes)  # 全连接层输出类别
        )

    def forward(self, x): # 定义ViT的forward方法，用于前向传播
        # 将输入图像转换为patch嵌入
        x = self.to_patch_embedding(x)
        x = x.transpose(1, 2)
        
        # 添加分类token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # 添加位置编码
        x = x + self.pos_embedding
        # 应用dropout
        x = self.dropout(x)
        
        # 通过Transformer块
        x = self.transformer(x)
        # 使用分类token进行最终分类
        x = self.mlp_head(x[:, 0])
        return x



if __name__ == "__main__":

    train_losses = []
    test_accuracies = []

    # 创建ViT模型实例
    # 针对CIFAR-10数据集调整参数：
    # - 图像尺寸32x32
    # - patch大小4x4
    # - 类别数10
    # - 嵌入维度512
    # - 6个Transformer块
    # - 8个注意力头
    # - MLP维度2048
    model = ViT(image_size=32, patch_size=4, num_classes=10, dim=512, depth=6, heads=8, mlp_dim=2048)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.03)  # AdamW优化器，带权重衰减
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)  # 余弦退火学习率调度器

    # 数据增强和预处理
    # 官方统计值
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std  = (0.2470, 0.2435, 0.2616)

    # 2. 预处理
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),      # 随机水平翻转
        transforms.RandomCrop(32, padding=4),   # 随机裁剪并填充
        transforms.ToTensor(),                  # 转为张量
        transforms.Normalize(                   # 标准化
            cifar10_mean,
            cifar10_std
        )
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            cifar10_mean,
            cifar10_std
        )
    ])

    # 3. 加载 CIFAR-10 数据集
    trainset = torchvision.datasets.CIFAR10(
        root='D:/CIFAR10',
        train=True,
        download=True,
        transform=train_transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root='D:/CIFAR10',
        train=False,
        download=True,
        transform=test_transform   # 注意这里是 test_transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=False, num_workers=2
    )
    # 设置设备（GPU或CPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device) # 将模型移动到GPU或CPU

    print(f"using {device}...")

    num_train_batches = len(trainloader)

    # 训练循环
    for epoch in range(100):  # 训练100轮
        model.train()  # 设置为训练模式
        running_loss = 0.0
        epoch_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data # 获取训练集数据
            inputs, labels = inputs.to(device), labels.to(device) # 将数据移动到GPU或CPU

            # 前向传播
            optimizer.zero_grad() # 梯度清零
            outputs = model(inputs) # 前向传播
            loss = criterion(outputs, labels) # 计算损失

            # 反向传播
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step() # 更新参数

            # 打印训练信息
            running_loss += loss.item() # 累加损失
            epoch_loss += loss.item()
            if i % 100 == 99:  # 每100个batch打印一次
                print(f"[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 100:.3f}")
                running_loss = 0.0

        # 更新学习率
        scheduler.step()

        # 记录训练可视化数据
        train_losses.append(epoch_loss / num_train_batches)

        # 在测试集上评估模型
        model.eval()  # 设置为评估模式
        correct = 0
        total = 0
        with torch.no_grad():  # 不计算梯度
            for data in testloader: # 获取测试集数据
                inputs, labels = data # 获取测试集数据
                inputs, labels = inputs.to(device), labels.to(device) # 将数据移动到GPU或CPU
                outputs = model(inputs) # 前向传播
                _, predicted = outputs.max(1) # 获取预测结果
                total += labels.size(0) # 计算总样本数
                correct += predicted.eq(labels).sum().item() # 计算正确样本数

        print(f"Epoch {epoch + 1} - Accuracy: {100 * correct / total:.2f}%")

        # 记录测试可视化数据
        acc = 100 * correct / total
        test_accuracies.append(acc)

    # 保存模型权重
    torch.save(model.state_dict(), 'vit_cifar10.pth')
    print("Model saved to vit_cifar10.pth")

    # 训练结束后，画图
    plt.figure(figsize=(10,4))

    # Loss 曲线
    plt.subplot(1,2,1)
    plt.plot(range(1, len(train_losses)+1), train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.title('Training Loss')

    # Accuracy 曲线
    plt.subplot(1,2,2)
    plt.plot(range(1, len(test_accuracies)+1), test_accuracies)
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy')

    plt.tight_layout()
    plt.show()
