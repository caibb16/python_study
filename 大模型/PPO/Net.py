import torch
import torch.nn as nn
import torch.nn.functional as F

# 策略网络，输出动作的概率分布，用于直接指导智能体如何在不同的状态下选择动作
class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # 对第二次的输出使用softmax函数，得到每个动作的概率分布
        x = F.softmax(self.fc2(x), dim=1)
        return x
    
# 价值网络，为智能体提供关于不同状态和动作的价值信息
class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        # 定义第二层全连接层，输出维度为1，表示状态价值
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# 优势函数，用于衡量在给定状态下采取某个动作相对于平均水平能好多少
def compute_advantage(gamma, lambda_, td_delta):
    # 将td_delta从张量转换为NumPy数组，方便后续计算
    td_delta = td_delta.detach().numpy()
    # 初始化优势值和优势列表
    advantage_list = []
    advantage = 0.0
    # 逆序遍历td_delta中的每个元素
    for delta in reversed(td_delta):
        # 计算每个时间步的优势值，添加到列表中
        advantage = gamma * lambda_ * advantage + delta
        advantage_list.append(advantage)
    # 将优势列表反转，恢复原始顺序
    advantage_list.reverse()
    # 将优势列表转换为张量并返回
    return torch.tensor(advantage_list, dtype=torch.float)