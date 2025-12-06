import torch
import numpy as np
from ROVConfig import ROVConfig, eul2rotm
from BlueROVDynamics import BlueROVDynamics
from ParallelESO import ParallelESO
from GeometricPIDBase import GeometricPIDBase


class ResidualROVEnv:
    def __init__(self, num_envs=1024, device='cuda:0'):
        self.device = device
        self.num_envs = num_envs
        self.cfg = ROVConfig(num_envs, device)

        self.plant = BlueROVDynamics(self.cfg)
        self.eso = ParallelESO(self.cfg, num_envs, device)
        self.base_ctrl = GeometricPIDBase(self.cfg)

        self.state = torch.zeros(num_envs, 12, device=device)
        self.last_thrust = torch.zeros(num_envs, 8, device=device)
        self.time = 0.0

        # Data Holders
        self.est_dist_body = torch.zeros(num_envs, 6, device=device)
        self.est_state = torch.zeros(num_envs, 12, device=device)

        self.last_action_rl = torch.zeros(num_envs, 6, device=device)

        # === [新增] 滤波器状态 ===
        # 1. 传感器输入滤波器状态 (Position/Attitude/Velocity/Omega)
        self.lpf_state_meas = torch.zeros(num_envs, 12, device=device)

        # 2. ESO 输出扰动滤波器状态
        self.lpf_dist_est = torch.zeros(num_envs, 6, device=device)

        # === [新增] 滤波系数 (Alpha) ===
        # alpha = dt / (dt + RC)
        # alpha 越小，滤波越强，滞后越大；alpha=1.0 为不滤波

        # 输入滤波系数 (可以稍微给大一点，保留动态)
        self.alpha_meas = 0.6

        # 输出滤波系数 (可以给小一点，平滑扰动)
        self.alpha_dist = 0.2

    def reset(self):
        state_ref_0 = self.get_reference_trajectory(0.0)
        self.state = state_ref_0.clone()

        # 开启随机化 (训练时)
        self.cfg.m = self.cfg.base_m * (0.8 + 0.4 * torch.rand(self.num_envs, 1, device=self.device))
        base_D = torch.tensor([4.03, 6.22, 5.18, 0.07, 0.07, 0.07], device=self.device)
        self.cfg.D_lin = base_D * (0.6 + 0.8 * torch.rand(self.num_envs, 6, device=self.device))

        self.last_thrust.fill_(0.0)
        self.last_action_rl.fill_(0.0)
        self.time = 0.0

        self.base_ctrl.reset()

        # === 复位滤波器 ===
        meas = self.sensor_model(self.state)
        self.lpf_state_meas = meas.clone()  # 初始化为当前测量值
        self.lpf_dist_est.fill_(0.0)

        self.eso.reset(meas[:, 0:6])

        return self.get_observation(state_ref_0)

    def step(self, action_rl_residual):
        state_ref = self.get_reference_trajectory(self.time)

        # 1. 传感器原始读数 (含高噪)
        raw_meas = self.sensor_model(self.state)

        # 2. [第一级] 输入低通滤波 (Input LPF)
        # y_k = (1-alpha)*y_{k-1} + alpha*u_k
        self.lpf_state_meas = (1.0 - self.alpha_meas) * self.lpf_state_meas + \
                              self.alpha_meas * raw_meas

        # 使用滤波后的测量值给 ESO 和 PID
        clean_meas = self.lpf_state_meas

        # 3. ESO 更新 (使用干净的测量值)
        raw_dist, eso_internal_state = self.eso.update(clean_meas[:, 0:6], self.last_thrust)

        # 4. [第二级] 输出低通滤波 (Output LPF)
        self.lpf_dist_est = (1.0 - self.alpha_dist) * self.lpf_dist_est + \
                            self.alpha_dist * raw_dist

        self.est_dist_body = self.lpf_dist_est

        # 状态估计组合：位置姿态用测量滤波值，速度用ESO估计值(通常比微分测量的更准)
        self.est_state = torch.cat([clean_meas[:, 0:6], eso_internal_state[:, 6:12]], dim=1)

        # 5. PID 计算 (使用滤波后的状态)
        tau_pid = self.base_ctrl.compute_force(self.est_state, state_ref)

        # 6. 力融合
        tau_rl = action_rl_residual * self.cfg.rl_scale
        tau_total = tau_pid + tau_rl

        # 7. 执行
        thrust_cmd = torch.matmul(self.cfg.T_pinv, tau_total.unsqueeze(-1)).squeeze(-1)
        thrust_cmd = torch.clamp(thrust_cmd, self.cfg.prop_min, self.cfg.prop_max)
        actual_thrust = self.motor_dynamics(thrust_cmd, self.last_thrust)
        self.last_thrust = actual_thrust

        # 8. 物理环境
        real_env_force = self.calc_ocean_current(self.state)
        self.state = self.plant.dynamics_step(self.state, actual_thrust, real_env_force)
        self.time += self.cfg.dt

        # 9. 奖励
        obs = self.get_observation(state_ref)

        err_p = torch.norm(self.state[:, 0:3] - state_ref[:, 0:3], dim=1)
        err_att = torch.norm(self.state[:, 3:6] - state_ref[:, 3:6], dim=1)
        err_v = torch.norm(self.state[:, 6:9] - state_ref[:, 6:9], dim=1)

        r_track = -1.0 * err_p - 2.0 * err_att - 0.1 * err_v
        total_power_proxy = torch.sum(torch.square(thrust_cmd), dim=1)
        r_energy = -0.001 * total_power_proxy
        r_reg = -0.001 * torch.norm(action_rl_residual, dim=1)

        action_diff = torch.norm(action_rl_residual - self.last_action_rl, dim=1)
        r_smooth = -0.1 * action_diff
        self.last_action_rl = action_rl_residual.clone()

        reward = r_track + r_energy + r_smooth + r_reg
        done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        return obs, reward, done, {}

    # ... (get_reference_trajectory, get_observation 等其他函数保持不变) ...
    # 记得保留完整的 get_reference_trajectory (Stunt版)

    def get_reference_trajectory(self, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=self.device)
        if t.ndim == 0:
            t = t.expand(self.num_envs)
        ref = torch.zeros(self.num_envs, 12, device=self.device)
        forward_speed = 0.3;
        radius = 1.5;
        omega = 0.4;
        depth0 = 2.0
        phi_amp = np.deg2rad(60);
        phi_freq = 0.6
        theta_amp = np.deg2rad(45);
        theta_freq = 0.4
        psi_amp = np.deg2rad(30);
        psi_freq = 0.2
        ref[:, 0] = forward_speed * t
        ref[:, 1] = radius * torch.sin(omega * t)
        ref[:, 2] = depth0 + radius * torch.cos(omega * t)
        ref[:, 6] = forward_speed
        ref[:, 7] = radius * omega * torch.cos(omega * t)
        ref[:, 8] = -radius * omega * torch.sin(omega * t)
        phi = phi_amp * torch.sin(phi_freq * t)
        theta = theta_amp * torch.sin(theta_freq * t)
        psi = psi_amp * torch.sin(psi_freq * t)
        ref[:, 3] = phi;
        ref[:, 4] = theta;
        ref[:, 5] = psi
        dphi = phi_amp * phi_freq * torch.cos(phi_freq * t)
        dtheta = theta_amp * theta_freq * torch.cos(theta_freq * t)
        dpsi = psi_amp * psi_freq * torch.cos(psi_freq * t)
        sph, cph = torch.sin(phi), torch.cos(phi)
        sth, cth = torch.sin(theta), torch.cos(theta)
        ref[:, 9] = dphi - dpsi * sth
        ref[:, 10] = dtheta * cph + dpsi * sph * cth
        ref[:, 11] = -dtheta * sph + dpsi * cph * cth
        return ref

    def get_observation(self, state_ref):
        e_p = state_ref[:, 0:3] - self.est_state[:, 0:3]
        e_att = state_ref[:, 3:6] - self.est_state[:, 3:6]
        e_v = state_ref[:, 6:9] - self.est_state[:, 6:9]
        e_w = state_ref[:, 9:12] - self.est_state[:, 9:12]
        return torch.cat([e_p, e_att, e_v, e_w, self.est_dist_body, self.est_state], dim=1)

    def sensor_model(self, state):
        noise = torch.randn_like(state) * self.cfg.sens_noise_sigma
        bias = torch.zeros_like(state)
        bias[:, 9:12] = self.cfg.sens_gyro_bias
        return state + noise + bias

    def motor_dynamics(self, cmd, prev):
        alpha = self.cfg.dt / self.cfg.prop_tc
        thrust = prev + alpha * (cmd - prev)
        return thrust

    def calc_ocean_current(self, state):
        return torch.zeros_like(state[:, 0:6])