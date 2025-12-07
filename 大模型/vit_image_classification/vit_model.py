"""
Vision Transformer (ViT) 模型实现
基于 "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange


class PatchEmbedding(nn.Module):
    """将图像分割成patches并进行embedding"""
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # 使用卷积层将图像分割成patches并进行线性变换
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e h w -> b (h w) e'),
        )
    
    def forward(self, x):
        x = self.projection(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力机制"""
    def __init__(self, embed_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim必须能被num_heads整除"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attention_dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.projection_dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.shape
        
        # 生成Q, K, V
        qkv = self.qkv(x)  # (batch_size, num_tokens, embed_dim * 3)
        qkv = rearrange(qkv, 'b n (three h d) -> three b h n d', 
                       three=3, h=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 计算注意力分数
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        
        # 加权求和
        attention_output = torch.matmul(attention_weights, v)
        attention_output = rearrange(attention_output, 'b h n d -> b n (h d)')
        
        # 输出投影
        output = self.projection(attention_output)
        output = self.projection_dropout(output)
        
        return output


class MLP(nn.Module):
    """前馈神经网络"""
    def __init__(self, embed_dim=256, hidden_dim=1024, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class TransformerEncoderBlock(nn.Module):
    """Transformer编码器块"""
    def __init__(self, embed_dim=256, num_heads=8, hidden_dim=1024, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.mlp = MLP(embed_dim, hidden_dim, dropout)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        # 注意力层 + 残差连接
        x = x + self.attention(self.layer_norm1(x))
        # MLP层 + 残差连接
        x = x + self.mlp(self.layer_norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer完整模型"""
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4,
        dropout=0.1,
        attention_dropout=0.1
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # CLS token (分类token)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # 位置编码
        self.position_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer编码器层
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                hidden_dim=int(embed_dim * mlp_ratio),
                dropout=attention_dropout
            )
            for _ in range(depth)
        ])
        
        # 分类头
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Patch Embedding
        x = self.patch_embed(x)  # (batch_size, num_patches, embed_dim)
        
        # 添加CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # 添加位置编码
        x = x + self.position_embedding
        x = self.dropout(x)
        
        # Transformer编码器
        for block in self.transformer_blocks:
            x = block(x)
        
        # 提取CLS token的输出用于分类
        x = self.layer_norm(x)
        cls_token_final = x[:, 0]
        
        # 分类
        logits = self.head(cls_token_final)
        
        return logits


def create_vit_small(num_classes=10, img_size=32):
    """创建小型ViT模型（适合4060 Laptop）"""
    return VisionTransformer(
        img_size=img_size,
        patch_size=4,
        in_channels=3,
        num_classes=num_classes,
        embed_dim=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4,
        dropout=0.1,
        attention_dropout=0.1
    )


def create_vit_tiny(num_classes=10, img_size=32):
    """创建微型ViT模型（更快的训练速度）"""
    return VisionTransformer(
        img_size=img_size,
        patch_size=4,
        in_channels=3,
        num_classes=num_classes,
        embed_dim=192,
        depth=4,
        num_heads=4,
        mlp_ratio=4,
        dropout=0.1,
        attention_dropout=0.1
    )


if __name__ == "__main__":
    # 测试模型
    model = create_vit_small(num_classes=10, img_size=32)
    x = torch.randn(4, 3, 32, 32)
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
