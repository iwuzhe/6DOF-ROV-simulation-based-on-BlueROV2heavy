import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from ResidualROVEnv import ResidualROVEnv


# === 1. 定义网络架构 (与训练时一致) ===
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
        return self.actor(x)


# === 2. 核心对比脚本 ===
def run_comparison():
    # --- 配置 ---
    MODEL_PATH = "ppo_rov_agent_final.pth"  # <--- 请修改为你最好的模型文件名
    DEVICE = 'cpu'  # 测试用 CPU 足够
    TOTAL_TIME = 60.0  # 测试时长

    # 初始化环境
    env = ResidualROVEnv(num_envs=1, device=DEVICE)

    # === 关键：制造“模型失配” (Model Mismatch) ===
    # 为了展示 RL 和 ESO 的优势，我们故意让真实物理参数与 PID 里的标称参数不一样
    print("配置测试环境：引入 20% 质量误差和 30% 阻尼误差...")

    # 备份标称参数（控制器以为的参数）
    nominal_m = env.cfg.base_m
    nominal_D = env.cfg.D_lin.clone()

    # 修改真实物理参数（环境实际运行的参数）
    # 假设真实 ROV 比模型重 20%，阻尼大 30%
    env.cfg.m = nominal_m * 1.2 * torch.ones(1, 1, device=DEVICE)
    env.cfg.D_lin = nominal_D * 1.3

    # 注意：PID 控制器内部仍使用 env.cfg 中的参数计算，
    # 但由于 GeometricPIDBase 初始化时可能拷贝了参数，我们需要确认 PID 是否受影响。
    # 实际上 GeometricPIDBase 是每一帧读 env.cfg 或者初始化读一次。
    # 查看代码，PID 参数是初始化固定的，但重力前馈用的 env.cfg.W。
    # 这里我们假设 PID 参数(Kp,Kd)固定，但前馈项会因为我们改了 cfg 而“作弊”变准。
    # 为了更严谨，我们应该只改 dynamics 里的参数，不改 cfg。
    # 但由于代码结构限制，这里改了 cfg 会同时影响 PID 的前馈。
    # **为了让对比更明显，我们手动把 PID 的积分项关小一点（模拟弱 PID），或者接受这个设定：**
    # 即使前馈准了，PID 的 Kp/Kd 针对的是原质量，现在质量变了，响应也会变差。

    # 加载 RL 模型
    agent = None
    if os.path.exists(MODEL_PATH):
        obs_dim = 30
        act_dim = 6
        agent = ActorCritic(obs_dim, act_dim).to(DEVICE)
        agent.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        agent.eval()
        print("RL 模型加载成功！")
    else:
        print(f"警告：找不到 {MODEL_PATH}，将跳过 RL 测试。")

    # 定义三种控制器模式
    modes = ['PID Only', 'PID + ESO', 'PID + RL']
    results = {}

    for mode in modes:
        if mode == 'PID + RL' and agent is None:
            continue

        print(f"正在运行测试: {mode} ...")

        # 重置环境
        obs = env.reset()
        # 再次强制覆盖参数（防止 reset 里的随机化覆盖我们的设置）
        env.cfg.m = nominal_m * 1.2 * torch.ones(1, 1, device=DEVICE)
        env.cfg.D_lin = nominal_D * 1.3

        # 数据记录
        log = {
            'time': [],
            'pos_err': [],
            'att_err': [],
            'thrust_norm': []
        }

        steps = int(TOTAL_TIME / env.cfg.dt)

        for step in range(steps):
            t = env.time

            # --- 动作选择 ---
            if mode == 'PID Only':
                # 纯 PID：残差动作为 0
                action = torch.zeros(1, 6, device=DEVICE)

            elif mode == 'PID + ESO':
                # PID + ESO：利用 ESO 估计的扰动进行前馈补偿
                # 原理：tau_total = tau_pid + action * scale
                # 我们希望 tau_total = tau_pid - est_dist
                # 所以 action * scale = -est_dist  =>  action = -est_dist / scale
                # 注意：est_dist_body 是上一帧的估计，作为近似
                dist = env.est_dist_body
                action = -dist / env.cfg.rl_scale

            elif mode == 'PID + RL':
                # PID + RL：使用网络预测
                with torch.no_grad():
                    action = agent.get_action_and_value(obs)

            # 执行一步
            obs, _, _, _ = env.step(action)

            # 获取参考轨迹用于计算误差
            state_ref = env.get_reference_trajectory(t)

            # 记录误差
            pos_act = env.state[0, 0:3]
            pos_ref = state_ref[0, 0:3]
            att_act = env.state[0, 3:6]
            att_ref = state_ref[0, 3:6]

            # 简单的欧氏距离误差
            e_p = torch.norm(pos_act - pos_ref).item()
            e_a = torch.norm(att_act - att_ref).item()  # 弧度

            # 记录推力消耗
            thrust_val = torch.norm(env.last_thrust).item()

            log['time'].append(t)
            log['pos_err'].append(e_p)
            log['att_err'].append(np.degrees(e_a))  # 转为度数
            log['thrust_norm'].append(thrust_val)

        results[mode] = log

    # === 3. 绘图与分析 ===
    plot_comparison(results)


def plot_comparison(results):
    plt.rcParams['axes.unicode_minus'] = False
    modes = list(results.keys())
    colors = {'PID Only': 'gray', 'PID + ESO': 'blue', 'PID + RL': 'red'}
    styles = {'PID Only': '--', 'PID + ESO': '-.', 'PID + RL': '-'}

    # 1. 误差曲线对比
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    # 位置误差
    for mode in modes:
        t = results[mode]['time']
        err = results[mode]['pos_err']
        rmse = np.sqrt(np.mean(np.square(err)))
        axs[0].plot(t, err, label=f"{mode} (RMSE={rmse:.3f}m)",
                    color=colors.get(mode), linestyle=styles.get(mode))

    axs[0].set_title('Position Tracking Error')
    axs[0].set_ylabel('Error [m]')
    axs[0].legend()
    axs[0].grid(True)

    # 姿态误差
    for mode in modes:
        t = results[mode]['time']
        err = results[mode]['att_err']
        rmse = np.sqrt(np.mean(np.square(err)))
        axs[1].plot(t, err, label=f"{mode} (RMSE={rmse:.3f}deg)",
                    color=colors.get(mode), linestyle=styles.get(mode))

    axs[1].set_title('Attitude Tracking Error')
    axs[1].set_ylabel('Error [deg]')
    axs[1].legend()
    axs[1].grid(True)

    # 推力功耗
    for mode in modes:
        t = results[mode]['time']
        thrust = results[mode]['thrust_norm']
        avg_thrust = np.mean(thrust)
        axs[2].plot(t, thrust, label=f"{mode} (Avg={avg_thrust:.1f}N)",
                    color=colors.get(mode), linestyle=styles.get(mode), alpha=0.8)

    axs[2].set_title('Thruster Usage (Energy)')
    axs[2].set_ylabel('Total Thrust [N]')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

    # 2. 柱状图统计对比
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ['Pos RMSE [m]', 'Att RMSE [deg]', 'Avg Thrust [N]']

    pos_rmses = [np.sqrt(np.mean(np.square(results[m]['pos_err']))) for m in modes]
    att_rmses = [np.sqrt(np.mean(np.square(results[m]['att_err']))) for m in modes]
    avg_thrusts = [np.mean(results[m]['thrust_norm']) for m in modes]

    x = np.arange(len(modes))

    ax[0].bar(x, pos_rmses, color=['gray', 'blue', 'red'], alpha=0.7)
    ax[0].set_xticks(x);
    ax[0].set_xticklabels(modes);
    ax[0].set_title(metrics[0])

    ax[1].bar(x, att_rmses, color=['gray', 'blue', 'red'], alpha=0.7)
    ax[1].set_xticks(x);
    ax[1].set_xticklabels(modes);
    ax[1].set_title(metrics[1])

    ax[2].bar(x, avg_thrusts, color=['gray', 'blue', 'red'], alpha=0.7)
    ax[2].set_xticks(x);
    ax[2].set_xticklabels(modes);
    ax[2].set_title(metrics[2])

    fig.suptitle('Performance Metrics Comparison')
    plt.show()


if __name__ == "__main__":
    run_comparison()