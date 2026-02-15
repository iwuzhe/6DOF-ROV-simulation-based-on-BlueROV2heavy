import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 解决OpenMP库冲突

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 导入你的模块
from ResidualROVEnv import ResidualROVEnv

def run_pid_test():
    # 1. 配置环境
    # 使用 num_envs=1 以便进行单条轨迹分析
    # 对于单条轨迹测试，CPU通常比GPU更快（无启动开销）
    device = 'cpu'
    print(f"Running PID Test on {device}...")

    env = ResidualROVEnv(num_envs=1, device=device)
    obs = env.reset()

    # 2. 仿真参数
    total_time = 100.0  # 仿真时长 (秒)
    steps = int(total_time / env.cfg.dt)

    # 3. 数据日志 (用于绘图)
    log_time = []
    log_pos_act = []  # 实际位置
    log_pos_ref = []  # 参考位置
    log_att_act = []  # 实际姿态
    log_att_ref = []  # 参考姿态
    log_thrust = []   # 推力指令

    print(f"Starting simulation for {total_time} seconds ({steps} steps)...")

    # 4. 主循环
    for step in range(steps):
        # === 关键点：强制 RL 动作为 0 ===
        # 这样 tau_total = tau_pid + 0
        action_zero = torch.zeros(1, 6, device=device)

        # 获取当前的参考轨迹 (为了记录数据，必须在 step 之前获取)
        current_time = env.time
        state_ref = env.get_reference_trajectory(current_time)

        # 记录数据 (转为 numpy CPU)
        log_time.append(current_time)
        log_pos_act.append(env.state[0, 0:3].cpu().numpy())
        log_pos_ref.append(state_ref[0, 0:3].cpu().numpy())
        log_att_act.append(env.state[0, 3:6].cpu().numpy())
        log_att_ref.append(state_ref[0, 3:6].cpu().numpy())

        # 执行环境步进
        obs, reward, done, info = env.step(action_zero)

        # 记录推力 (查看是否饱和)
        # last_thrust 是 [num_envs, 8]，取第0个环境
        log_thrust.append(env.last_thrust[0].cpu().numpy())

    print("Simulation finished. Plotting results...")

    # 5. 数据转换
    log_time = np.array(log_time)
    log_pos_act = np.array(log_pos_act)
    log_pos_ref = np.array(log_pos_ref)
    log_att_act = np.array(log_att_act)
    log_att_ref = np.array(log_att_ref)
    log_thrust = np.array(log_thrust)

    # 6. 绘图
    plot_results(log_time, log_pos_act, log_pos_ref, log_att_act, log_att_ref, log_thrust)

def plot_results(t, pos_act, pos_ref, att_act, att_ref, thrust_cmd):
    # 设置字体，防止中文乱码 (可选)
    plt.rcParams['axes.unicode_minus'] = False

    # --- 图1: 3D 轨迹对比 ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(pos_ref[:, 0], pos_ref[:, 1], pos_ref[:, 2], 'g--', label='Reference', linewidth=2)
    ax.plot(pos_act[:, 0], pos_act[:, 1], pos_act[:, 2], 'b-', label='PID Actual', linewidth=1.5)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('3D Trajectory Tracking')
    ax.legend()
    plt.show()

    # --- 图2: 位置跟踪曲线 XYZ ---
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    labels = ['X', 'Y', 'Z']
    for i in range(3):
        axs[i].plot(t, pos_ref[:, i], 'g--', label='Ref')
        axs[i].plot(t, pos_act[:, i], 'b-', label='Act')
        axs[i].set_ylabel(f'{labels[i]} [m]')
        axs[i].grid(True)
        axs[i].legend(loc='upper right')
    axs[-1].set_xlabel('Time [s]')
    fig.suptitle('Position Tracking')
    plt.show()

    # --- 图3: 姿态跟踪曲线 RPY ---
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    labels = ['Roll', 'Pitch', 'Yaw']
    for i in range(3):
        # 转换为度数
        axs[i].plot(t, np.degrees(att_ref[:, i]), 'g--', label='Ref')
        axs[i].plot(t, np.degrees(att_act[:, i]), 'b-', label='Act')
        axs[i].set_ylabel(f'{labels[i]} [deg]')
        axs[i].grid(True)
        axs[i].legend(loc='upper right')
    axs[-1].set_xlabel('Time [s]')
    fig.suptitle('Attitude Tracking')
    plt.show()

    # --- 图4: 推力指令 (检查饱和) ---
    fig, ax = plt.subplots(figsize=(12, 6))
    # 绘制8个电机的曲线
    for i in range(8):
        ax.plot(t, thrust_cmd[:, i], label=f'Prop {i+1}', linewidth=1.0)
    
    # 画出 +/- 50N 的限制线
    ax.axhline(y=50, color='r', linestyle='--', linewidth=2, label='Max Limit')
    ax.axhline(y=-40, color='r', linestyle='--', linewidth=2, label='Min Limit')
    
    ax.set_ylabel('Thrust [N]')
    ax.set_xlabel('Time [s]')
    ax.set_title('Thruster Output (Check Saturation)')
    ax.grid(True)
    # 图例放外面防止遮挡
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_pid_test()