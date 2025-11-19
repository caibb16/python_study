import torch as th
import numpy as np
import torch.nn.functional as F
from Net import PolicyNet, ValueNet, compute_advantage

class PPO:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lambda_, epochs, eps, gamma, device):
        # 初始化Actor和Critic网络
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        # 设置对应的优化器
        self.actor_optimizer = th.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = th.optim.Adam(self.critic.parameters(), lr=critic_lr)
        # 设置强化学习的参数
        self.gamma = gamma
        self.lambda_ = lambda_
        self.epochs = epochs
        self.eps = eps  # 截断范围
        self.device = device

    def take_action(self, state):
        # 通过Actor网络获取动作的概率分布
        state = th.tensor(np.array([state]), dtype=th.float).to(self.device)
        probs = self.actor(state)
        # 从动作的概率分布中采样一个动作
        action_dist = th.distributions.Categorical(probs)
        action = action_dist.sample()  # 概率越大的动作被采样到的概率越大
        return action.item()  # 返回action的值，值是某个动作的索引
    
    def update(self, transition_dict):
        # 获取转换的状态、动作、奖励、下一次状态和结束标志
        # 先转换为 numpy 数组，再转为张量（避免从列表直接转换的性能警告）
        states = th.tensor(np.array(transition_dict['states']), dtype=th.float).to(self.device)
        actions = th.tensor(np.array(transition_dict['actions'])).view(-1, 1).to(self.device)
        rewards = th.tensor(np.array(transition_dict['rewards']), dtype=th.float).view(-1, 1).to(self.device)
        next_states = th.tensor(np.array(transition_dict['next_states']), dtype=th.float).to(self.device)
        dones = th.tensor(np.array(transition_dict['dones']), dtype=th.float).view(-1, 1).to(self.device)

        # 计算TD目标和TD误差（TD：Temporal Difference）
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)  # td_target是在 t + 1 时刻做出的预测，包含事实成分
        td_delta = td_target - self.critic(states)  # self.critic(states)是在 t 时刻做出的预测，无事实成分，二者相减为TD误差

        # 计算优势函数
        advantage = compute_advantage(self.gamma, self.lambda_, td_delta.cpu()).to(self.device)
        # 计算旧策略的动作概率,PPO更新过程中旧策略保持不变
        old_log_probs = th.log(self.actor(states).gather(1, actions)).detach()

        # 进行多次迭代更新Actor和Critic网络
        for _ in range(self.epochs):
            # 计算新策略的动作概率
            log_probs = th.log(self.actor(states).gather(1, actions))
            # 计算概率比率，用来衡量新老策略在同一动作上的偏移程度
            ratio = th.exp(log_probs - old_log_probs)

            # 计算截断的目标函数
            surr1 = ratio * advantage
            surr2 = th.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage  # clamp函数用于限制ratio的范围在[1-eps, 1+eps]之间

            # 计算Actor和Critic的损失
            actor_loss = th.mean(-th.min(surr1, surr2))
            critic_loss = th.mean(F.mse_loss(self.critic(states), td_target.detach()))

            # 清空优化器梯度
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            # 反向传播
            actor_loss.backward()
            critic_loss.backward()

            # 更新参数
            self.actor_optimizer.step()
            self.critic_optimizer.step()
