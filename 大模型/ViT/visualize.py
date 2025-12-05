import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib

# 1. 指定一个支持中文的字体（Windows 下通常有 SimHei、Microsoft YaHei）
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体
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
    
    def forward_with_attn(self, x):
        x = self.to_patch_embedding(x)
        x = x.transpose(1, 2)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        x = self.dropout(x)

        attn_maps = []
        for block in self.transformer:
            x, attn = block(x, return_attn=True)
            attn_maps.append(attn)  # [B, num_heads, num_patches+1]
        return x, attn_maps

# CIFAR-10 类别名
cifar10_classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

   
# 可视化函数
def visualize_attention(model, image_path, device):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    transform = test_transform
    input_tensor = transform(image).unsqueeze(0).to(device)

    # 获取预测分类
    with torch.no_grad():
        outputs = model(input_tensor)
        predicted_class = outputs.argmax(dim=1).item()
        predicted_label = cifar10_classes[predicted_class]

    # 获取 attention
    _, attn_maps = model.forward_with_attn(input_tensor)

    # 根据 attn_maps 的维度来取 cls→patch 的注意力
    last_attn = attn_maps[-1]  # 可能是 4D，也可能是 3D

    attn = last_attn  # 直接用
    cls_attn = attn[:, 0, 1:]


    # 多头平均
    avg_attn = cls_attn.mean(dim=0)

    grid_size = int(model.num_patches ** 0.5)
    attn_map = avg_attn.reshape(grid_size, grid_size).cpu().detach().numpy()

    # 显示图像
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].imshow(image)
    axs[0].set_title(f"此图片被模型识别为: {predicted_label}")
    axs[0].axis("off")

    axs[1].imshow(image)
    axs[1].imshow(attn_map, cmap='jet', alpha=0.5,
                  extent=(0, image.size[0], image.size[1], 0))
    axs[1].set_title("Attention Map展示")
    axs[1].axis("off")

    plt.suptitle(image_path)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

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

    # 数据增强和预处理
    # 官方统计值
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std  = (0.2470, 0.2435, 0.2616)

    # 2. 预处理
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            cifar10_mean,
            cifar10_std
        )
    ])

    # 设置设备（GPU或CPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载已训练模型
    model.load_state_dict(torch.load("vit_cifar10.pth"))
    model.forward_with_attn = ViT.forward_with_attn.__get__(model)
    model.to(device)

    # 可视化指定图片 attention
    image_paths = [
        "custom_images/cat1107.png",
        "custom_images/cat1129.png",
        "custom_images/cat1205.png",
        "custom_images/cat1206.png",
        "custom_images/cat1212.png",
        "custom_images/cat1633.png",
        "custom_images/airplane105.png",
        "custom_images/airplane165.png",
        "custom_images/airplane168.png",
        "custom_images/airplane170.png",
        "custom_images/airplane352.png",
        "custom_images/airplane370.png"
    ]

    for path in image_paths:
        visualize_attention(model, path, device)
