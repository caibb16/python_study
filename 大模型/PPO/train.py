import torch as th
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm
from PPO_Model import PPO
from average import moving_average

def train_on_policy_agent(env, agent, num_episodes):
    return_list = []  # 用于存储每个周期的总回报
    for i in range(10):  # 总共进行10次迭代
        with tqdm(total=int(num_episodes / 10), desc=f"Iteration: {i}") as pbar:  # 每次迭代创建一个进度条
            for i_episode in range(int(num_episodes / 10)):  # 每个迭代周期中，智能体进行num_episodes/10个周期的训练
                # 在这里添加与环境交互的代码
                episode_return = 0  # 用于存储当前周期的总回报
                transition_dict = {'states': [], 'actions': [],'next_states': [], 'rewards': [],  'dones': []}  # 存储转换数据
                state, _ = env.reset()  # 重置环境，获取初始状态
                done = False
                while not done:  # 采样阶段，获取当前周期的所有转换数据，直到周期结束
                    action = agent.take_action(state)  # 选择动作
                    # 兼容 Gym 与 Gymnasium 的 step 返回
                    _step_out = env.step(action)
                    if isinstance(_step_out, tuple) and len(_step_out) == 5:
                        next_state, reward, terminated, truncated, _ = _step_out
                        done = bool(terminated or truncated)
                    else:
                        next_state, reward, done, _ = _step_out
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state  # 更新状态
                    episode_return += reward  # 更新总回报
                return_list.append(episode_return)  # 记录当前周期的总回报
                agent.update(transition_dict)  # 学习阶段，通过学习这个周期的状态、动作和奖励等信息来更新策略
                if (i_episode + 1) % 10 == 0:  # 每10个周期更新一次进度条
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),  # 输出当前周期数
                                     'return': '%.3f' % np.mean(return_list[-10:])})  # 输出最近10个周期的平均回报
                pbar.update(1)
    return return_list

actor_lr = 1e-3   # Actor网络的学习率
critic_lr = 1e-2   # Critic网络的学习率
num_episodes = 300   # 训练的总周期数
hidden_dim = 128   # 隐藏层的维度
gamma = 0.98   # 折扣因子
lambda_ = 0.95   # GAE的lambda参数
epochs = 5   # 更新的轮数
eps = 0.2   # PPO的剪切范围
device = th.device("cuda" if th.cuda.is_available() else "cpu")
env_name = "CartPole-v1"   # 环境名称
env = gym.make(env_name)   # 创建环境
state_dim = env.observation_space.shape[0]   # 状态的维度
action_dim = env.action_space.n   # 动作的维度

# 实例化智能体
agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lambda_, epochs, eps, gamma, device)
# 训练智能体
return_list = train_on_policy_agent(env, agent, num_episodes)

episodes_list = list(range(len(return_list)))
mv_return_list = moving_average(return_list, 9)
plt.plot(episodes_list, mv_return_list)
plt.xlabel('Episodes')
plt.ylabel("Returns")
plt.title(f"PPO on {env_name}")
plt.show()