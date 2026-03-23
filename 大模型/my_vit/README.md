# Vision Transformer (ViT) 图像识别项目

基于Transformer架构实现的Vision Transformer模型，用于CIFAR-10图像分类任务。

## 项目简介

本项目从零实现了Vision Transformer（ViT）模型，包含：
- ✅ 完整的ViT模型架构（Patch Embedding、Multi-Head Attention、Transformer Encoder等）
- ✅ CIFAR-10数据集加载和数据增强
- ✅ 完整的训练和验证流程
- ✅ 模型测试和可视化
- ✅ 针对RTX 4060 Laptop优化的模型配置

## 文件结构

```
vit_image_classification/
├── vit_model.py          # ViT模型实现
├── train.py              # 训练脚本
├── test.py               # 测试和推理脚本
├── requirements.txt      # 依赖包列表
├── README.md            # 项目说明
├── data/                # 数据集目录（自动下载）
└── checkpoints/         # 模型检查点目录（训练时创建）
```

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (用于GPU加速)
- 其他依赖见 requirements.txt

## 安装依赖

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## 快速开始

### 1. 训练模型

```bash
python train.py
```

训练参数说明：
- `model_type`: 模型大小，可选 `'small'` 或 `'tiny'`（默认：`'small'`）
- `num_epochs`: 训练轮次（默认：100）
- `batch_size`: 批次大小（默认：128）
- `learning_rate`: 学习率（默认：3e-4）
- `weight_decay`: 权重衰减（默认：0.1）

训练过程中会：
- 自动下载CIFAR-10数据集
- 定期保存模型检查点
- 保存最佳模型到 `checkpoints/best_model.pth`
- 记录训练历史到 `checkpoints/training_history.json`

### 2. 测试模型

```bash
python test.py
```

测试脚本会：
- 加载训练好的最佳模型
- 在测试集上评估性能
- 生成预测可视化图像（`predictions.png`）
- 显示各类别的准确率统计

### 3. 模型推理（单张图像）

修改 `test.py` 末尾的代码：

```python
# 预测单张图像
predict_single_image(model, 'your_image.jpg', device=device)
```

## 模型架构

### ViT-Small（推荐用于4060 Laptop）
- Embedding维度：256
- 编码器层数：6
- 注意力头数：8
- MLP倍率：4
- 参数量：约2.7M

### ViT-Tiny（更快的训练速度）
- Embedding维度：192
- 编码器层数：4
- 注意力头数：4
- MLP倍率：4
- 参数量：约1.1M

## 主要特性

### 1. Patch Embedding
将32×32图像分割成4×4的patches（总共64个patches），每个patch转换为256维向量。

### 2. Multi-Head Self-Attention
8头注意力机制，允许模型关注图像的不同部分和不同特征。

### 3. Transformer Encoder
6层Transformer编码器块，每层包含：
- Layer Normalization
- Multi-Head Attention
- 残差连接
- MLP（前馈网络）

### 4. 分类头
使用CLS token的最终输出进行分类。

### 5. 数据增强
- 随机裁剪（padding=4）
- 随机水平翻转
- 标准化（CIFAR-10的均值和标准差）

### 6. 训练优化
- AdamW优化器（权重衰减）
- 余弦退火学习率调度
- 混合精度训练支持（可选）

## 性能预期

在CIFAR-10数据集上：
- ViT-Small：约85-90%准确率（100 epochs）
- ViT-Tiny：约82-87%准确率（100 epochs）

训练时间（RTX 4060 Laptop）：
- ViT-Small：约2-3小时（100 epochs）
- ViT-Tiny：约1-2小时（100 epochs）

## CIFAR-10类别

1. 飞机 (airplane)
2. 汽车 (automobile)
3. 鸟 (bird)
4. 猫 (cat)
5. 鹿 (deer)
6. 狗 (dog)
7. 青蛙 (frog)
8. 马 (horse)
9. 船 (ship)
10. 卡车 (truck)

## 技术细节

### Patch Embedding实现
使用卷积层实现patch分割和线性变换：
```python
nn.Conv2d(in_channels=3, out_channels=embed_dim, 
          kernel_size=patch_size, stride=patch_size)
```

### 位置编码
使用可学习的位置编码，在训练过程中自动优化。

### CLS Token
在序列开头添加特殊的分类token，用于最终的分类任务。

## 常见问题

### Q: 显存不足怎么办？
A: 降低batch_size或使用ViT-Tiny模型。

### Q: 训练速度慢？
A: 检查是否启用了GPU加速，或降低模型大小。

### Q: 准确率不理想？
A: 增加训练轮次，调整学习率，或使用更大的模型。

## 参考文献

- Dosovitskiy, A., et al. (2020). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." ICLR 2021.
- Vaswani, A., et al. (2017). "Attention is All You Need." NeurIPS 2017.


