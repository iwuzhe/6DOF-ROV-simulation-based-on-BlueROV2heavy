import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
import matplotlib.pyplot as plt
from ResidualROVEnv import ResidualROVEnv


def run_eso_test():
    device = 'cpu'
    print(f"正在测试 ESO 性能 (Device: {device})...")

    # 1. 初始化环境
    env = ResidualROVEnv(num_envs=1, device=device)
    obs = env.reset()

    # 确保 ESO 重置
    env.eso.reset(env.sensor_model(env.state)[:, 0:6])

    # 2. 测试参数
    total_time = 20.0  # 测试 5 秒
    steps = int(total_time / env.cfg.dt)

    # 定义一个已知的人造扰动 (模拟洋流或外力)
    # 在 t = 1.0s 时，在 X 轴施加 +20N 的力，在 Yaw 轴施加 +5Nm 的力矩
    # 这是 Body Frame 下的力
    known_disturbance = torch.zeros(1, 6, device=device)

    # 日志
    log_t = []
    log_force_true = []
    log_force_est = []
    log_pos_true = []
    log_pos_meas = []
    log_pos_est = []
    log_vel_true = []
    log_vel_est = []

    print("开始仿真...")
    for step in range(steps):
        t = env.time

        # --- A. 施加阶跃扰动 ---
        if t > 1.0:
            known_disturbance[0, 0] = 40.0  # Fx = 20N
            known_disturbance[0, 5] = 5.0  # Tz = 5Nm

        # --- B. 运行环境 (开环或弱 PID 维持位置) ---
        # 我们这里用纯 PID 维持姿态，以免跑太远
        state_ref = env.get_reference_trajectory(t)
        # 强制参考轨迹静止在原点，方便观察
        state_ref.fill_(0.0)

        # 获取传感器数据
        obs_meas = env.sensor_model(env.state)

        # --- C. 核心：运行 ESO ---
        # 注意：ESO 也就是在这里计算 est_dist_body
        dist_force_est, eso_state = env.eso.update(obs_meas[:, 0:6], env.last_thrust)

        # --- D. 物理步进 ---
        # PID 计算控制力
        tau_pid = env.base_ctrl.compute_force(obs_meas, state_ref)
        thrust_cmd = torch.matmul(env.cfg.T_pinv, tau_pid.unsqueeze(-1)).squeeze(-1)
        thrust_cmd = torch.clamp(thrust_cmd, env.cfg.prop_min, env.cfg.prop_max)
        actual_thrust = env.motor_dynamics(thrust_cmd, env.last_thrust)
        env.last_thrust = actual_thrust

        # [关键] 将“已知扰动”注入物理环境
        # dynamics_step 接收的 ocean_current_force 是我们要测试的真值
        # 注意：BlueROVDynamics 里的 ocean_current_force 是加在 RHS 上的，方向定义为外力
        env.state = env.plant.dynamics_step(env.state, actual_thrust, ocean_current_force=known_disturbance)
        env.time += env.cfg.dt

        # --- E. 记录数据 ---
        log_t.append(t)

        # 1. 扰动对比
        log_force_true.append(known_disturbance[0].numpy().copy())
        log_force_est.append(dist_force_est[0].numpy().copy())

        # 2. 状态对比 (位置 X)
        log_pos_true.append(env.state[0, 0].item())  # 真实 X
        log_pos_meas.append(obs_meas[0, 0].item())  # 传感器 X (含噪)
        log_pos_est.append(eso_state[0, 0].item())  # ESO 估计 X

        # 3. 速度对比 (速度 U)
        log_vel_true.append(env.state[0, 6].item())  # 真实 U
        # 注意：ESO 估计的速度是在 Earth Frame 还是 Body Frame?
        # ParallelESO 中的 z2 是 Earth Frame 速度。
        # 而 env.state[6] 是 Body Frame 速度 (u)。
        # 为了对比，我们需要把 ESO 的 Earth 速度转回 Body，或者把真值转到 Earth。
        # 这里简单起见，我们假设角度很小 (hover)，Earth X velocity ≈ Body u
        log_vel_est.append(eso_state[0, 6].item())

        # --- 绘图 ---
    plot_eso_test(log_t, log_force_true, log_force_est,
                  log_pos_true, log_pos_meas, log_pos_est,
                  log_vel_true, log_vel_est)


def plot_eso_test(t, f_true, f_est, p_true, p_meas, p_est, v_true, v_est):
    f_true = np.array(f_true)
    f_est = np.array(f_est)

    fig, axs = plt.subplots(3, 2, figsize=(12, 10))

    # 1. 扰动估计 - X轴力
    axs[0, 0].plot(t, f_true[:, 0], 'g--', label='True Disturbance (20N)', linewidth=2)
    axs[0, 0].plot(t, f_est[:, 0], 'r-', label='ESO Estimate', alpha=0.8)
    axs[0, 0].set_title('Disturbance Estimation (Force X)')
    axs[0, 0].set_ylabel('Force [N]')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # 2. 扰动估计 - Z轴力矩
    axs[0, 1].plot(t, f_true[:, 5], 'g--', label='True Disturbance (5Nm)', linewidth=2)
    axs[0, 1].plot(t, f_est[:, 5], 'r-', label='ESO Estimate', alpha=0.8)
    axs[0, 1].set_title('Disturbance Estimation (Torque Z)')
    axs[0, 1].set_ylabel('Torque [Nm]')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # 3. 位置估计 - X
    axs[1, 0].plot(t, p_true, 'k-', label='True Pos', linewidth=2)
    axs[1, 0].plot(t, p_meas, 'g.', markersize=2, label='Measured (Sensor)', alpha=0.3)
    axs[1, 0].plot(t, p_est, 'r--', label='ESO State z1')
    axs[1, 0].set_title('State Estimation (Position X)')
    axs[1, 0].set_ylabel('Pos [m]')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # 4. 速度估计
    axs[1, 1].plot(t, v_true, 'k-', label='True Vel (Body u)')
    axs[1, 1].plot(t, v_est, 'r--', label='ESO Est (Earth vx)')
    axs[1, 1].set_title('Velocity Estimation (Check Phase Lag)')
    axs[1, 1].set_ylabel('Vel [m/s]')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    # 5. 噪声分析 (放大看稳态)
    # 取最后 1 秒的数据计算噪声
    idx = int(len(t) * 0.8)
    noise_est = np.std(f_est[idx:, 0] - f_true[idx:, 0])
    axs[2, 0].plot(t[idx:], f_true[idx:, 0], 'g--')
    axs[2, 0].plot(t[idx:], f_est[idx:, 0], 'r-')
    axs[2, 0].set_title(f'Steady State Noise (Std Dev: {noise_est:.2f} N)')
    axs[2, 0].set_ylabel('Force [N]')
    axs[2, 0].grid(True)

    axs[2, 1].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_eso_test()