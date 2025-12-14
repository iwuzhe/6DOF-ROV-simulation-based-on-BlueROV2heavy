import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from ResidualROVEnv import ResidualROVEnv


# === 修正 1: 网络结构必须与 train_rl.py 完全一致 ===
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        # 使用 train_rl.py 中的结构: feature_extractor + actor_mean
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh()
        )
        self.actor_mean = nn.Linear(128, act_dim)
        self.actor_activation = nn.Tanh()

        # Critic 必须定义，否则加载权重时会报错 "Unexpected key: critic..."
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.log_std = nn.Parameter(torch.ones(1, act_dim) * -0.5)

    def get_action_and_value(self, x, action=None):
        hidden = self.feature_extractor(x)
        mean = self.actor_activation(self.actor_mean(hidden))
        return mean


def run_rl_test():
    # === 配置 ===
    MODEL_PATH = "ppo_rov_agent_stage2.pth"  # 确保文件名正确

    device = 'cpu'
    print(f"Loading model from {MODEL_PATH} on {device}...")

    # 1. 初始化环境
    # 注意：为了看到真实效果，建议去 ResidualROVEnv.py 临时把 limit_pos_error 改大
    # 或者在这里手动修改 env 属性（如果代码支持）
    env = ResidualROVEnv(num_envs=1, device=device, is_eval=True)
    obs = env.reset()

    # 2. 初始化网络
    obs_dim = 33
    act_dim = 6
    agent = ActorCritic(obs_dim, act_dim).to(device)

    # 准备归一化参数容器
    obs_mean = torch.zeros(obs_dim, device=device)
    obs_var = torch.ones(obs_dim, device=device)

    # === 修正 2: 正确加载权重字典 ===
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=device)

        # A. 提取 model_state_dict 加载
        try:
            agent.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError as e:
            print(f"权重加载失败，请检查网络定义是否与训练时一致。\n详细错误: {e}")
            return

        # B. 加载归一化参数 (Obs Norm)
        if 'obs_norm_mean' in checkpoint:
            obs_mean = checkpoint['obs_norm_mean']
            obs_var = checkpoint['obs_norm_var']
            print(">>> 模型及归一化参数加载成功！")
        else:
            print(">>> 警告：未找到归一化参数！Agent 可能会表现极差！")
    else:
        print(f"错误: 找不到文件 {MODEL_PATH}")
        return

    agent.eval()

    # 3. 仿真循环
    total_time = 100.0
    steps = int(total_time / env.cfg.dt)

    log_time = []
    log_pos_act, log_pos_ref = [], []
    log_att_act, log_att_ref = [], []
    log_thrust = []
    log_rl_action = []

    print(f"Starting simulation for {total_time} s...")

    for step in range(steps):
        current_time = env.time
        state_ref = env.get_reference_trajectory(current_time)

        # === 核心：归一化 + 推理 ===
        with torch.no_grad():
            # 手动归一化 (RunningMeanStd 的逻辑)
            norm_obs = (obs - obs_mean) / torch.sqrt(obs_var + 1e-8)
            action_rl = agent.get_action_and_value(norm_obs)

        # 执行动作
        next_obs, reward, done, info = env.step(action_rl)

        # 记录数据
        log_time.append(current_time)
        log_pos_act.append(env.state[0, 0:3].cpu().numpy())
        log_pos_ref.append(state_ref[0, 0:3].cpu().numpy())
        log_att_act.append(env.state[0, 3:6].cpu().numpy())
        log_att_ref.append(state_ref[0, 3:6].cpu().numpy())
        log_thrust.append(env.last_thrust[0].cpu().numpy())
        log_rl_action.append(action_rl[0].cpu().numpy())

        obs = next_obs

    print("Done. Plotting...")

    # 绘图部分
    t = np.array(log_time)
    pos_act = np.array(log_pos_act);
    pos_ref = np.array(log_pos_ref)
    att_act = np.array(log_att_act);
    att_ref = np.array(log_att_ref)
    thrust = np.array(log_thrust);
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

    # 2. 位置跟踪
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    labels = ['X', 'Y', 'Z']
    for i in range(3):
        axs[i].plot(t, pos_ref[:, i], 'g--', label='Ref')
        axs[i].plot(t, pos_act[:, i], 'b-', label='Act')
        axs[i].set_ylabel(f'{labels[i]} [m]')
        axs[i].grid(True)
    axs[0].legend()
    fig.suptitle('Position Tracking')
    plt.show()

    # 3. 姿态跟踪
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

    # 4. RL 输出
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


if __name__ == "__main__":
    run_rl_test()