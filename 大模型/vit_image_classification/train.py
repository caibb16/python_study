"""
训练Vision Transformer模型
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import time
import os

from vit_model import create_vit_small, create_vit_tiny


def get_cifar10_loaders(batch_size=128, num_workers=4):
    """获取CIFAR-10数据加载器"""
    
    # 数据增强
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # 加载数据集
    train_dataset = datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=train_transform
    )
    
    test_dataset = datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=True, 
        transform=test_transform
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [训练]')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # 更新进度条
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, test_loader, criterion, device, epoch):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc=f'Epoch {epoch} [验证]')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
    
    epoch_loss = running_loss / len(test_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def train_vit(
    model_type='small',
    num_epochs=100,
    batch_size=128,
    learning_rate=3e-4,
    weight_decay=0.1,
    device='cuda',
    save_dir='./checkpoints'
):
    """训练ViT模型"""
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    if device.type == 'cuda':
        print(f'GPU型号: {torch.cuda.get_device_name(0)}')
        print(f'显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
    
    # 创建模型
    if model_type == 'small':
        model = create_vit_small(num_classes=10, img_size=32)
    elif model_type == 'tiny':
        model = create_vit_tiny(num_classes=10, img_size=32)
    else:
        raise ValueError("model_type必须是'small'或'tiny'")
    
    model = model.to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f'模型类型: ViT-{model_type}')
    print(f'总参数量: {total_params:,}')
    
    # 获取数据加载器
    print('\n加载CIFAR-10数据集...')
    train_loader, test_loader = get_cifar10_loaders(batch_size=batch_size)
    print(f'训练样本数: {len(train_loader.dataset)}')
    print(f'测试样本数: {len(test_loader.dataset)}')
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # 学习率调度器（余弦退火）
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # 训练循环
    best_acc = 0.0
    train_history = []
    
    print(f'\n开始训练 (总共{num_epochs}个epochs)...\n')
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # 验证
        val_loss, val_acc = validate(model, test_loader, criterion, device, epoch)
        
        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录历史
        train_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': current_lr
        })
        
        # 打印汇总信息
        print(f'\nEpoch {epoch}/{num_epochs}:')
        print(f'  训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
        print(f'  验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')
        print(f'  学习率: {current_lr:.6f}\n')
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'model_type': model_type
            }
            torch.save(checkpoint, os.path.join(save_dir, 'best_model.pth'))
            print(f'✓ 保存最佳模型 (验证准确率: {best_acc:.2f}%)\n')
        
        # 定期保存检查点
        if epoch % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'model_type': model_type
            }
            torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth'))
    
    # 训练结束
    total_time = time.time() - start_time
    print(f'\n训练完成！')
    print(f'总耗时: {total_time / 3600:.2f} 小时')
    print(f'最佳验证准确率: {best_acc:.2f}%')
    
    # 保存训练历史
    import json
    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        json.dump(train_history, f, indent=2)
    
    return model, train_history, best_acc


if __name__ == "__main__":
    # 训练ViT模型
    # 使用'tiny'模型以在4060 Laptop上获得更快的训练速度
    model, history, best_acc = train_vit(
        model_type='small',  # 可选: 'small' 或 'tiny'
        num_epochs=100,
        batch_size=128,
        learning_rate=3e-4,
        weight_decay=0.1,
        device='cuda',
        save_dir='./checkpoints'
    )
