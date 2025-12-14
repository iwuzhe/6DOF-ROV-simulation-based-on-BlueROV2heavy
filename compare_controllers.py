import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from ResidualROVEnv import ResidualROVEnv


# === 1. 修复 ActorCritic 定义 ===
# 必须与 train_rl.py 中的定义完全一致，才能正确加载权重
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        # 共享特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh()
        )

        # Actor 部分
        self.actor_mean = nn.Linear(128, act_dim)
        self.actor_activation = nn.Tanh()

        # Critic 部分 (必须定义，否则加载报错)
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(),
            nn.Linear(128, 1)
        )

        # Log Std (必须定义，否则加载报错)
        self.log_std = nn.Parameter(torch.ones(1, act_dim) * -0.5)

    def get_action_and_value(self, x):
        """
        测试专用推理函数：
        只返回确定性的动作均值 (Deterministic Action)，不采样。
        """
        hidden = self.feature_extractor(x)
        mean = self.actor_activation(self.actor_mean(hidden))
        return mean


def run_comparison():
    # 确保文件名与你训练保存的阶段一致 (Stage 1 或 Final)
    # 如果刚跑完 Stage 1，这里可能是 "ppo_rov_agent_stage1.pth" 或者 "ppo_rov_agent_final.pth"
    MODEL_PATH = "ppo_rov_agent_40.pth"
    DEVICE = 'cpu'  # 测试用 CPU 足够
    TEST_DURATION = 60.0

    print(f"初始化环境 (Device: {DEVICE})...")
    # is_eval=True 会让环境尽量使用标称参数，但下面我们会手动覆盖
    env = ResidualROVEnv(num_envs=1, device=DEVICE, is_eval=True)

    # === 2. 制造严重的模型失配 (用于测试鲁棒性) ===
    print("应用模型失配参数...")
    nominal_m = env.cfg_nominal.base_m
    nominal_D = env.cfg_nominal.D_lin.clone()

    # 模拟真实环境的变化
    # A. 质量增加 20% (约 3kg)
    env.cfg_true.m = nominal_m * 1.2 * torch.ones(1, 1, device=DEVICE)
    env.cfg_true.W = env.cfg_true.m * env.cfg_true.g

    # B. 阻力增加 50% (比如挂了海生物)
    env.cfg_true.D_lin = nominal_D * 1.5

    # C. 附加质量变化 (影响科里奥利力)
    env.cfg_true.M_added = env.cfg_nominal.M_added.unsqueeze(0) * 1.2

    # D. 推进器效率下降
    env.thruster_efficiency = 0.9 * torch.ones(1, 8, device=DEVICE)

    # === 3. 加载模型 ===
    obs_dim = 33
    act_dim = 6
    agent = ActorCritic(obs_dim, act_dim).to(DEVICE)

    # 归一化参数容器
    obs_mean = torch.zeros(obs_dim, device=DEVICE)
    obs_var = torch.ones(obs_dim, device=DEVICE)

    if os.path.exists(MODEL_PATH):
        print(f"加载模型: {MODEL_PATH}")
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

        # 加载网络权重
        agent.load_state_dict(checkpoint['model_state_dict'])

        # 加载归一化参数 (非常重要！)
        if 'obs_norm_mean' in checkpoint:
            obs_mean = checkpoint['obs_norm_mean']
            obs_var = checkpoint['obs_norm_var']
            print("归一化参数加载成功！")
        else:
            print("警告：未找到归一化参数，性能可能会很差！")
    else:
        print(f"错误：找不到模型文件 {MODEL_PATH}，跳过 RL 测试。")
        agent = None

    modes = ['PID Only', 'PID + ESO', 'PID + RL']
    results = {}

    for mode in modes:
        if mode == 'PID + RL' and agent is None: continue
        print(f"正在运行测试: {mode} ...")

        # 重置环境
        obs = env.reset()

        # 重新应用失配 (因为 reset 可能会重置部分随机参数)
        env.cfg_true.m = nominal_m * 1.2 * torch.ones(1, 1, device=DEVICE)
        env.cfg_true.W = env.cfg_true.m * env.cfg_true.g
        env.cfg_true.D_lin = nominal_D * 1.5
        env.cfg_true.M_added = env.cfg_nominal.M_added.unsqueeze(0) * 1.2
        env.thruster_efficiency = 0.9 * torch.ones(1, 8, device=DEVICE)

        log = {'time': [], 'pos_err': [], 'thrust_norm': [], 'att_err': []}
        steps = int(TEST_DURATION / env.cfg.dt)

        for step in range(steps):
            t = env.time

            # --- 动作选择 ---
            if mode == 'PID Only':
                action = torch.zeros(1, 6, device=DEVICE)  # 无残差

            elif mode == 'PID + ESO':
                # 简单的 ESO 前馈补偿 (假设 RL Scale 对应力矩)
                # 这种简单的直接抵消通常效果一般，因为没有调优
                dist = env.est_dist_body
                action = -dist / env.cfg.rl_scale

            elif mode == 'PID + RL':
                with torch.no_grad():
                    # 1. 归一化观测
                    norm_obs = (obs - obs_mean) / torch.sqrt(obs_var + 1e-8)
                    # 2. 推理 (确定性)
                    action = agent.get_action_and_value(norm_obs)

            # --- 环境步进 ---
            # 注意：ResidualROVEnv 的 step 接收的是残差动作
            obs, _, _, _ = env.step(action)

            # --- 记录数据 ---
            state_ref = env.get_reference_trajectory(t)

            # 位置误差 (欧氏距离)
            e_p = torch.norm(env.state[0, 0:3] - state_ref[0, 0:3]).item()
            # 姿态误差 (Roll/Pitch 这种关键指标)
            e_att = torch.norm(env.state[0, 3:6] - state_ref[0, 3:6]).item()
            # 推力消耗
            thrust = torch.norm(env.last_thrust).item()

            log['time'].append(t)
            log['pos_err'].append(e_p)
            log['att_err'].append(e_att)
            log['thrust_norm'].append(thrust)

        results[mode] = log

    plot_comparison(results)


def plot_comparison(results):
    plt.rcParams['axes.unicode_minus'] = False
    modes = list(results.keys())
    colors = {'PID Only': 'gray', 'PID + ESO': 'blue', 'PID + RL': 'red'}

    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # 1. 位置误差
    for mode in modes:
        t = results[mode]['time']
        data = results[mode]['pos_err']
        rmse = np.sqrt(np.mean(np.square(data)))
        axs[0].plot(t, data, label=f"{mode} (RMSE={rmse:.3f}m)", color=colors.get(mode))
    axs[0].set_ylabel('Position Error [m]')
    axs[0].set_title('Comparison: Position Tracking')
    axs[0].legend()
    axs[0].grid(True)

    # 2. 姿态误差
    for mode in modes:
        t = results[mode]['time']
        data = np.degrees(results[mode]['att_err'])  # 转为度
        rmse = np.sqrt(np.mean(np.square(data)))
        axs[1].plot(t, data, label=f"{mode} (RMSE={rmse:.2f} deg)", color=colors.get(mode))
    axs[1].set_ylabel('Attitude Error [deg]')
    axs[1].set_title('Comparison: Attitude Tracking')
    axs[1].legend()
    axs[1].grid(True)

    # 3. 推力消耗
    for mode in modes:
        t = results[mode]['time']
        data = results[mode]['thrust_norm']
        avg = np.mean(data)
        axs[2].plot(t, data, label=f"{mode} (Avg={avg:.1f}N)", color=colors.get(mode), alpha=0.6)
    axs[2].set_ylabel('Total Thrust [N]')
    axs[2].set_xlabel('Time [s]')
    axs[2].set_title('Comparison: Control Effort')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_comparison()