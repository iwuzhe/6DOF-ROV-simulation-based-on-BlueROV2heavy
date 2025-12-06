import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from ResidualROVEnv import ResidualROVEnv


# === 1. 定义网络架构 (必须与训练时完全一致) ===
# 为了方便，这里直接复制了 train_rl.py 中的定义
# 如果你在 train_rl.py 中修改了网络，这里也必须同步修改
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(),
            nn.Linear(128, act_dim), nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.log_std = nn.Parameter(torch.zeros(1, act_dim))

    def get_action_and_value(self, x, action=None):
        mean = self.actor(x)
        # 测试时通常直接使用均值(mean)作为确定性动作，不加噪声
        # 也可以保留噪声来看看鲁棒性，这里我们用确定性策略
        return mean


def run_rl_test():
    # === 配置 ===
    # 请修改这里为你训练出来的文件名！
    MODEL_PATH = "ppo_rov_agent_final.pth"  # <--- 修改这里！！！

    device = 'cpu'  # 测试时用 CPU 即可
    print(f"Loading model from {MODEL_PATH} on {device}...")

    # 1. 初始化环境
    env = ResidualROVEnv(num_envs=1, device=device)
    obs = env.reset()

    # 2. 初始化网络并加载权重
    obs_dim = 30
    act_dim = 6
    agent = ActorCritic(obs_dim, act_dim).to(device)

    try:
        agent.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Model loaded successfully!")
    except FileNotFoundError:
        print(f"错误: 找不到文件 {MODEL_PATH}。请检查文件名是否正确，或者是否还没训练完。")
        return

    agent.eval()  # 切换到评估模式

    # 3. 仿真循环
    total_time = 100.0
    steps = int(total_time / env.cfg.dt)

    log_time = []
    log_pos_act, log_pos_ref = [], []
    log_att_act, log_att_ref = [], []
    log_thrust = []
    log_rl_action = []  # 记录 RL 输出的力矩

    print(f"Starting simulation for {total_time} s...")

    for step in range(steps):
        # 获取参考轨迹用于记录
        current_time = env.time
        state_ref = env.get_reference_trajectory(current_time)

        # === 核心：使用 RL Agent 计算动作 ===
        with torch.no_grad():
            # obs 是 [1, 30] 的张量
            # 输出 action 是 [1, 6] 的张量
            action_rl = agent.get_action_and_value(obs)

        # 执行动作
        next_obs, reward, done, info = env.step(action_rl)

        # === 记录数据 ===
        log_time.append(current_time)
        log_pos_act.append(env.state[0, 0:3].cpu().numpy())
        log_pos_ref.append(state_ref[0, 0:3].cpu().numpy())
        log_att_act.append(env.state[0, 3:6].cpu().numpy())
        log_att_ref.append(state_ref[0, 3:6].cpu().numpy())
        log_thrust.append(env.last_thrust[0].cpu().numpy())
        log_rl_action.append(action_rl[0].cpu().numpy())

        obs = next_obs

    print("Done. Plotting...")

    # 转换为 numpy
    t = np.array(log_time)
    pos_act = np.array(log_pos_act);
    pos_ref = np.array(log_pos_ref)
    att_act = np.array(log_att_act);
    att_ref = np.array(log_att_ref)
    thrust = np.array(log_thrust)
    rl_act = np.array(log_rl_action)

    plot_results(t, pos_act, pos_ref, att_act, att_ref, thrust, rl_act)


def plot_results(t, pos_act, pos_ref, att_act, att_ref, thrust, rl_act):
    plt.rcParams['axes.unicode_minus'] = False

    # 1. 3D 轨迹
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(pos_ref[:, 0], pos_ref[:, 1], pos_ref[:, 2], 'g--', label='Ref')
    ax.plot(pos_act[:, 0], pos_act[:, 1], pos_act[:, 2], 'b-', label='RL+PID')
    ax.set_title('3D Trajectory (RL Controlled)')
    ax.legend()
    plt.show()

    # 2. 轨迹跟踪
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    labels = ['X', 'Y', 'Z']
    for i in range(3):
        axs[i].plot(t, np.degrees(pos_ref[:, i]), 'g--', label='Ref')
        axs[i].plot(t, np.degrees(pos_act[:, i]), 'b-', label='Act')
        axs[i].set_ylabel(f'{labels[i]} [m]')
        axs[i].grid(True)
    axs[0].legend()
    fig.suptitle('position Tracking')
    plt.show()

    # 2. 姿态跟踪
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    labels = ['Roll', 'Pitch', 'Yaw']
    for i in range(3):
        axs[i].plot(t, np.degrees(att_ref[:, i]), 'g--', label='Ref')
        axs[i].plot(t, np.degrees(att_act[:, i]), 'b-', label='Act')
        axs[i].set_ylabel(f'{labels[i]} [deg]')
        axs[i].grid(True)
    axs[0].legend()
    fig.suptitle('Attitude Tracking')
    plt.show()

    # 3. RL 输出动作 (残差力矩)
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(t, rl_act[:, 0:3])
    axs[0].set_ylabel('RL Force [N]')
    axs[0].legend(['Fx', 'Fy', 'Fz'])
    axs[0].grid(True)

    axs[1].plot(t, rl_act[:, 3:6])
    axs[1].set_ylabel('RL Torque [Nm]')
    axs[1].legend(['Tx', 'Ty', 'Tz'])
    axs[1].grid(True)
    fig.suptitle('RL Agent Output (Residual)')
    plt.show()

    # 4. 推力
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t, thrust)
    ax.set_ylabel('Thruster PWM/Force')
    ax.set_title('Thruster Output')
    ax.grid(True)
    plt.show()


if __name__ == "__main__":
    run_rl_test()