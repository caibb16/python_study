# Transformer 日期序列转换项目

## 项目简介

本项目实现了基于 Transformer 架构的日期序列转换模型，支持中英文日期格式的相互转换。项目包含数据生成、模型训练、评估和推理等完整流程，适合学习和实验 Transformer 结构在序列到序列任务中的应用。

## 主要模块

- `train.py`：模型训练主脚本，包含数据加载、批量训练、损失打印和模型保存。
- `evaluate.py`：模型评估与推理脚本，支持单条样本的预测输出。
- `Transformer.py`：Transformer 主体，包括编码器、解码器、嵌入层、掩码机制等核心实现。
- `MulHdAtt.py`：多头注意力机制实现，包含头分割、缩放点积注意力等细节。
- `PosEmb.py`：位置编码实现，支持正余弦位置嵌入。
- `Encoder.py` / `Decoder.py`：编码器和解码器结构定义。
- `DateDataset.py`：日期数据集生成与处理，支持批量样本生成和索引转换。

## 快速开始

1. 安装依赖  
   推荐使用 Python 3.8+，需安装 PyTorch 和 numpy：
   ```
   pip install torch numpy
   ```

2. 训练模型  
   ```
   python train.py
   ```
   训练完成后会自动保存模型参数到 `transformer.pth`。

3. 评估模型  
   ```
   python evaluate.py
   ```
   可查看模型对测试样本的预测效果。

## 主要功能说明

- 支持自定义数据集大小和批量训练。
- 采用多头注意力和位置编码，提升模型表达能力。
- 自动处理填充和掩码，保证训练和推理的正确性。
- 训练过程自动打印损失，便于观察模型收敛情况。

## 文件结构

```
Transformer/
├── train.py
├── evaluate.py
├── Transformer.py
├── MulHdAtt.py
├── PosEmb.py
├── Encoder.py
├── Decoder.py
├── DateDataset.py
├── README.md
```

## 注意事项

- 默认填充 token 索引为 2，请确保数据和模型参数一致。
- 若使用 GPU，需安装对应版本的 PyTorch，并自动检测设备。
- 训练和评估脚本可根据实际需求调整参数，如批量大小、序列长度等。

## 致谢

本项目参考了原始 Transformer 论文及相关 PyTorch 实现，适合教学、实验和小型应用场景。