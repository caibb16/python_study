"""
测试和推理脚本
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import random

from vit_model import create_vit_small, create_vit_tiny

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# CIFAR-10类别名称
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

CIFAR10_CLASSES_CN = [
    '飞机', '汽车', '鸟', '猫', '鹿',
    '狗', '青蛙', '马', '船', '卡车'
]


def load_model(checkpoint_path, device='cuda'):
    """加载训练好的模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_type = checkpoint.get('model_type', 'small')
    
    # 创建模型
    if model_type == 'small':
        model = create_vit_small(num_classes=10, img_size=32)
    elif model_type == 'tiny':
        model = create_vit_tiny(num_classes=10, img_size=32)
    else:
        raise ValueError(f"未知的模型类型: {model_type}")
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f'成功加载模型: {checkpoint_path}')
    print(f'模型类型: ViT-{model_type}')
    print(f'训练轮次: {checkpoint["epoch"]}')
    if 'best_acc' in checkpoint:
        print(f'最佳准确率: {checkpoint["best_acc"]:.2f}%')
    
    return model


def test_model(model, test_loader, device='cuda'):
    """在测试集上评估模型"""
    model.eval()
    correct = 0
    total = 0
    
    # 每个类别的统计
    class_correct = [0] * 10
    class_total = [0] * 10
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='测试中')
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 统计每个类别的准确率
            for i in range(len(targets)):
                label = targets[i].item()
                class_correct[label] += (predicted[i] == targets[i]).item()
                class_total[label] += 1
            
            pbar.set_postfix({'acc': 100. * correct / total})
    
    # 总体准确率
    overall_acc = 100. * correct / total
    print(f'\n整体准确率: {overall_acc:.2f}% ({correct}/{total})')
    
    # 每个类别的准确率
    print('\n各类别准确率:')
    for i in range(10):
        if class_total[i] > 0:
            acc = 100. * class_correct[i] / class_total[i]
            print(f'  {CIFAR10_CLASSES_CN[i]:>4s} ({CIFAR10_CLASSES[i]:>10s}): {acc:.2f}% ({class_correct[i]}/{class_total[i]})')
    
    return overall_acc


def visualize_predictions(model, test_loader, device='cuda', num_images=16):
    """可视化模型预测结果"""
    model.eval()
    
    # 随机选择一批图像
    dataiter = iter(test_loader)
    skip_batches = random.randint(0, len(test_loader) - 1)
    for _ in range(skip_batches):
        next(dataiter)
    images, labels = next(dataiter)
    images = images[:num_images]
    labels = labels[:num_images]
    
    # 预测
    with torch.no_grad():
        outputs = model(images.to(device))
        _, predicted = outputs.max(1)
        predicted = predicted.cpu()
    
    # 反归一化用于显示
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    images = images * std + mean
    images = torch.clamp(images, 0, 1)
    
    # 绘图
    rows = int(np.sqrt(num_images))
    cols = int(np.ceil(num_images / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.5))
    axes = axes.flatten()
    
    for idx in range(num_images):
        ax = axes[idx]
        
        # 显示图像
        img = images[idx].permute(1, 2, 0).numpy()
        ax.imshow(img)
        
        # 标题
        true_label = CIFAR10_CLASSES_CN[labels[idx]]
        pred_label = CIFAR10_CLASSES_CN[predicted[idx]]
        
        if labels[idx] == predicted[idx]:
            color = 'green'
            title = f'√ {pred_label}'
        else:
            color = 'red'
            title = f'× {pred_label}\n(真实: {true_label})'
        
        ax.set_title(title, color=color, fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions.png', dpi=150, bbox_inches='tight')
    print(f'\n预测可视化已保存到 predictions.png')
    plt.show()


def predict_single_image(model, image_path, device='cuda'):
    """预测单张图像"""
    from PIL import Image
    
    # 加载和预处理图像
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # 预测
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted_class = torch.max(probabilities, 0)
    
    # 显示结果
    print(f'\n预测结果:')
    print(f'  类别: {CIFAR10_CLASSES_CN[predicted_class]} ({CIFAR10_CLASSES[predicted_class]})')
    print(f'  置信度: {confidence.item() * 100:.2f}%')
    
    # 显示Top-3预测
    print(f'\nTop-3预测:')
    top3_prob, top3_classes = torch.topk(probabilities, 3)
    for i in range(3):
        cls_idx = top3_classes[i].item()
        prob = top3_prob[i].item()
        print(f'  {i+1}. {CIFAR10_CLASSES_CN[cls_idx]:>4s} ({CIFAR10_CLASSES[cls_idx]:>10s}): {prob*100:.2f}%')
    
    return predicted_class.item(), confidence.item()


def get_test_loader(batch_size=128):
    """获取测试数据加载器"""
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    test_dataset = datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=True, 
        transform=test_transform
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    return test_loader


if __name__ == "__main__":
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}\n')
    
    # 加载模型
    checkpoint_path = './checkpoints/best_model.pth'
    if not os.path.exists(checkpoint_path):
        print(f'错误: 找不到模型文件 {checkpoint_path}')
        print('请先运行 train.py 训练模型')
        exit(1)
    
    model = load_model(checkpoint_path, device=device)
    
    # 获取测试数据
    print('\n加载测试数据...')
    test_loader = get_test_loader(batch_size=128)
    
    # 测试模型
    print('\n开始测试模型...')
    test_acc = test_model(model, test_loader, device=device)
    
    # 可视化预测结果
    print('\n生成预测可视化...')
    visualize_predictions(model, test_loader, device=device, num_images=16)
    
    print('\n测试完成！')
