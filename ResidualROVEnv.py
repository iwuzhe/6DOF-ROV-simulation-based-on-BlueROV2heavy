import torch
import numpy as np
from ROVConfig import ROVConfig, eul2rotm
from BlueROVDynamics import BlueROVDynamics
from ParallelESO import ParallelESO
from GeometricPIDBase import GeometricPIDBase


class ResidualROVEnv:
    def __init__(self, num_envs=1024, device='cuda:0', is_eval=False):
        self.device = device
        self.num_envs = num_envs
        self.is_eval = is_eval

        # === 1. 双重配置 ===
        self.cfg_nominal = ROVConfig(num_envs, device)
        self.cfg_true = ROVConfig(num_envs, device)

        # === 2. 模块初始化 ===
        self.plant = BlueROVDynamics(self.cfg_true)
        self.eso = ParallelESO(self.cfg_nominal, num_envs, device)
        self.base_ctrl = GeometricPIDBase(self.cfg_nominal)
        self.cfg = self.cfg_nominal

        self.state = torch.zeros(num_envs, 12, device=device)
        self.last_thrust = torch.zeros(num_envs, 8, device=device)
        self.time = 0.0

        # Data Holders
        self.est_dist_body = torch.zeros(num_envs, 6, device=device)
        self.est_state = torch.zeros(num_envs, 12, device=device)
        self.last_action_rl = torch.zeros(num_envs, 6, device=device)

        # === [关键修复 1] 滤波器设置 ===

        # A. ESO 输出扰动滤波 (防止控制信号震荡)
        self.lpf_dist_est = torch.zeros(num_envs, 6, device=device)
        self.alpha_dist = 0.2  # 0.2 = 强滤波，平滑 ESO 输出

        # B. 速度滤波 (用于 ESO 输入)
        self.lpf_vel = torch.zeros(num_envs, 6, device=device)
        self.alpha_vel = 1.0  # 1.0 = 直通 (ESO 需要原始动态，或者给极轻微滤波)

        # C. [新增] 位置/姿态滤波 (解决 PID 抖动的关键!)
        # 传感器有噪声，高增益 PID 会放大噪声导致震荡。必须滤波。
        self.lpf_pos = torch.zeros(num_envs, 6, device=device)
        self.alpha_pos = 1.0  # 0.2 = 强滤波，滤掉高频噪声

    def reset(self):
        if self.is_eval:
            self.time = 0.0
        else:
            self.time = torch.rand(1, device=self.device).item() * 60.0

        state_ref_0 = self.get_reference_trajectory(self.time)

        if self.is_eval:
            self.state = state_ref_0.clone()
        else:
            init_noise = (torch.rand(self.num_envs, 12, device=self.device) - 0.5) * 0.1
            self.state = state_ref_0.clone() + init_noise

        # === 3. 物理参数随机化 ===
        base_D = torch.tensor([4.03, 6.22, 5.18, 0.07, 0.07, 0.07], device=self.device)

        if self.is_eval:
            self.cfg_true.m = self.cfg_nominal.base_m * torch.ones(self.num_envs, 1, device=self.device)
            self.cfg_true.W = self.cfg_true.m * self.cfg_true.g
            self.cfg_true.D_lin = base_D.repeat(self.num_envs, 1)
            self.thruster_efficiency = 1.0
        else:
            # 质量 +/- 20%
            self.cfg_true.m = self.cfg_nominal.base_m * (0.8 + 0.4 * torch.rand(self.num_envs, 1, device=self.device))
            self.cfg_true.W = self.cfg_true.m * self.cfg_true.g
            # 阻力随机化
            self.cfg_true.D_lin = base_D * (0.6 + 0.8 * torch.rand(self.num_envs, 6, device=self.device))
            # 附加质量随机化 +/- 20%
            base_Ma = self.cfg_nominal.M_added.unsqueeze(0).repeat(self.num_envs, 1)
            self.cfg_true.M_added = base_Ma * (0.8 + 0.4 * torch.rand(self.num_envs, 6, device=self.device))
            # 推力效率随机化 +/- 15%
            self.thruster_efficiency = 0.85 + 0.3 * torch.rand(self.num_envs, 8, device=self.device)  # 0.85 ~ 1.15
            # 惯性矩随机化 ±15%（模拟载荷变化、水中附加质量不确定性）
            base_I = torch.tensor([0.25, 0.35, 0.45], device=self.device)
            self.cfg_true.I = base_I * (0.85 + 0.3 * torch.rand(self.num_envs, 3, device=self.device))
            # 浮心位置随机化 ±2cm（模拟安装误差、配重调整）
            base_rB = torch.tensor([0.0, 0.0, -0.008], device=self.device)
            rB_noise = (torch.rand(self.num_envs, 3, device=self.device) - 0.5) * 0.02
            self.cfg_true.rB = base_rB + rB_noise
        # === 4. PID 复位 ===
        self.base_ctrl.reset()
        self.last_action_rl.fill_(0.0)

        # === 4. 动力学配平 ===
        equil_thrust = self._get_equilibrium_thrust(self.state)
        self.last_thrust = equil_thrust.clone()

        # === 5. 滤波器复位 ===
        meas = self.sensor_model(self.state)

        self.lpf_dist_est.fill_(0.0)
        self.lpf_vel = meas[:, 6:12].clone()
        self.lpf_pos = meas[:, 0:6].clone()  # [关键] 初始化姿态滤波器

        self.eso.reset(meas[:, 6:12])

        return self.get_observation(state_ref_0)

    def _get_equilibrium_thrust(self, state):
        nu = state[:, 6:12]
        eta = state[:, 0:6]
        D_vec = (self.cfg_true.D_lin + self.cfg_true.D_quad * torch.abs(nu)) * nu
        phi, theta = eta[:, 3], eta[:, 4]
        sth, cth = torch.sin(theta), torch.cos(theta)
        sph, cph = torch.sin(phi), torch.cos(phi)
        W, B_buoy = self.cfg_true.W, self.cfg_true.B

        g_vec = torch.zeros_like(nu)
        W_val = W.view(-1)
        g_vec[:, 0] = (W_val - B_buoy) * sth
        g_vec[:, 1] = -(W_val - B_buoy) * cth * sph
        g_vec[:, 2] = -(W_val - B_buoy) * cth * cph
        rB = self.cfg_true.rB
        g_vec[:, 3] = -(rB[:, 1] * B_buoy * cth * cph - rB[:, 2] * B_buoy * cth * sph)
        g_vec[:, 4] = -(rB[:, 2] * B_buoy * sth + rB[:, 0] * B_buoy * cth * cph)
        g_vec[:, 5] = -(rB[:, 0] * B_buoy * cth * sph - rB[:, 1] * B_buoy * sth)

        tau_eq = D_vec + g_vec
        thrust_cmd = torch.matmul(self.cfg_true.T_pinv, tau_eq.unsqueeze(-1)).squeeze(-1)
        thrust_cmd = torch.clamp(thrust_cmd, self.cfg_true.prop_min, self.cfg_true.prop_max)
        return thrust_cmd

    def step(self, action_rl_residual):
        state_ref = self.get_reference_trajectory(self.time)

        # 1. 传感器获取 (带噪声)
        raw_meas = self.sensor_model(self.state)

        # === [关键修复 2] 信号滤波处理 ===

        # A. 姿态滤波 (给 PID 用) - 消除 ±1° 抖动
        self.lpf_pos = (1.0 - self.alpha_pos) * self.lpf_pos + self.alpha_pos * raw_meas[:, 0:6]

        # B. 速度滤波 (给 ESO 用) - 保持较快响应
        self.lpf_vel = (1.0 - self.alpha_vel) * self.lpf_vel + self.alpha_vel * raw_meas[:, 6:12]

        # 组装 clean_meas
        clean_meas = raw_meas.clone()
        clean_meas[:, 0:6] = self.lpf_pos  # PID 看到的是平滑的位置
        clean_meas[:, 6:12] = self.lpf_vel  # ESO 看到的是(相对)原始的速度

        # 2. ESO 更新
        raw_dist, eso_internal_state = self.eso.update(
            clean_meas[:, 6:12],
            self.last_thrust,
            clean_meas[:, 3:6]
        )

        # 3. ESO 输出滤波
        self.lpf_dist_est = (1.0 - self.alpha_dist) * self.lpf_dist_est + self.alpha_dist * raw_dist
        self.est_dist_body = self.lpf_dist_est

        # 4. 控制计算
        # 组合状态：平滑位置 + ESO估计速度 (比差分速度更准)
        self.est_state = torch.cat([clean_meas[:, 0:6], eso_internal_state[:, 0:6]], dim=1)

        tau_pid = self.base_ctrl.compute_force(self.est_state, state_ref)
        tau_rl = action_rl_residual * self.cfg.rl_scale
        tau_total = tau_pid + tau_rl

        # 5. 执行
        thrust_cmd = torch.matmul(self.cfg_nominal.T_pinv, tau_total.unsqueeze(-1)).squeeze(-1)
        thrust_cmd = torch.clamp(thrust_cmd, self.cfg.prop_min, self.cfg.prop_max)
        actual_cmd = thrust_cmd * self.thruster_efficiency  # 乘上效率系数
        actual_thrust = self.motor_dynamics(actual_cmd, self.last_thrust)
        self.last_thrust = actual_thrust

        # 6. 物理步进
        real_env_force = self.calc_ocean_current(self.state)
        self.state = self.plant.dynamics_step(self.state, actual_thrust, real_env_force)
        self.time += self.cfg.dt

        # 7. 奖励计算 (使用修正后的系数)
        obs = self.get_observation(state_ref)

        err_p = torch.norm(self.state[:, 0:3] - state_ref[:, 0:3], dim=1)
        err_att = torch.norm(self.state[:, 3:6] - state_ref[:, 3:6], dim=1)
        err_v = torch.norm(self.state[:, 6:9] - state_ref[:, 6:9], dim=1)
        err_w = torch.norm(self.state[:, 9:12] - state_ref[:, 9:12], dim=1)
        # [关键修复 3] 降低奖励数值，防止 Loss 爆炸
        # 单步奖励控制在 0.5 ~ 1.5 之间
        r_pos = 0.5 * (1.0 - torch.tanh(2.0 * err_p))
        r_att = 0.5 * (1.0 - torch.tanh(12.0 * err_att))
        r_vel = 0.1 * (1.0 - torch.tanh(0.5 * err_v))
        r_omega = 0.2 * (1.0 - torch.tanh(1.0 * err_w))

        r_reg = -0.001 * torch.norm(action_rl_residual, dim=1)
        r_smooth = -0.01 * torch.norm(action_rl_residual - self.last_action_rl, dim=1)

        self.last_action_rl = action_rl_residual.clone()

        done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # === 严格的终止条件  ===
        # 1. 位置误差阈值：设定为 3.0 米
        # 2. 姿态误差阈值：设定为 1.0弧度 (约 3 度)
        limit_pos_error = 3.0
        limit_att_error = 1.0

        # 计算是否越界
        is_too_far = err_p > limit_pos_error
        is_unstable = (torch.abs(self.state[:, 3]) > limit_att_error) | \
                      (torch.abs(self.state[:, 4]) > limit_att_error)
        # 惩罚"位置和姿态同时超标"的状态（在done之前！）
        conflict_penalty = torch.zeros(self.num_envs, device=self.device)
        # 如果位置误差>0.3m 且 姿态误差>0.05rad(约3度)
        is_conflict = (err_p > 0.5) & (err_att > 0.09)
        conflict_penalty[is_conflict] = -1.0

        # 3. 产生 Done 信号
        done = is_too_far | is_unstable

        # === 奖励函数重构：生存奖励 ===
        r_survival = 1.0
        r_death = -10.0 * done.float()
        # 原有的误差奖励保持，作为引导信号 (Shaping Reward)
        reward = (r_pos + r_att + r_vel + r_omega + r_reg + r_smooth) * 0.1 + r_survival + r_death + conflict_penalty

        # === 自动复位===
        # 向量化环境必须能够自动复位死掉的那个环境，而不是全部重置
        if torch.any(done):
            with torch.no_grad():
                env_ids = torch.nonzero(done).squeeze()
                if env_ids.ndim == 0: env_ids = env_ids.unsqueeze(0)

                #  获取当前时刻的参考状态
                ref_now = self.get_reference_trajectory(self.time)

                #  将死掉的环境状态重置回参考状态
                self.state[env_ids] = ref_now[env_ids].clone()
                self.state[env_ids] += (torch.rand(len(env_ids), 12, device=self.device) - 0.5) * 0.05

                #  重置 PID 的积分项
                self.base_ctrl.int_e_pos[env_ids] = 0.0
                self.base_ctrl.int_e_att[env_ids] = 0.0

                # D. 重置滤波器和历史动作
                self.lpf_vel[env_ids] = self.sensor_model(self.state)[env_ids, 6:12]
                self.lpf_pos[env_ids] = self.sensor_model(self.state)[env_ids, 0:6]
                self.last_action_rl[env_ids] = 0.0
                self.last_thrust[env_ids] = self._get_equilibrium_thrust(self.state)[env_ids]


        return obs, reward, done, {}


    def get_observation(self, state_ref):
        e_p_earth = state_ref[:, 0:3] - self.est_state[:, 0:3]
        e_att = state_ref[:, 3:6] - self.est_state[:, 3:6]

        phi, theta, psi = self.est_state[:, 3], self.est_state[:, 4], self.est_state[:, 5]
        R = eul2rotm(phi, theta, psi)
        e_p_body = torch.bmm(R.transpose(1, 2), e_p_earth.unsqueeze(-1)).squeeze(-1)

        e_v_body = state_ref[:, 6:9] - self.est_state[:, 6:9]
        e_w_body = state_ref[:, 9:12] - self.est_state[:, 9:12]

        obs_e_p = torch.clamp(e_p_body, -2.0, 2.0)
        obs_e_att = e_att * 10.0
        obs_e_v = e_v_body
        obs_e_w = e_w_body
        obs_dist = self.est_dist_body / 10.0

        obs_state_partial = torch.cat([
            self.est_state[:, 3:6],
            self.est_state[:, 6:12]
        ], dim=1)

        obs_last_act = self.last_action_rl

        return torch.cat([
            obs_e_p, obs_e_att, obs_e_v, obs_e_w, obs_dist,
            obs_state_partial,
            obs_last_act
        ], dim=1)

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
        ref[:, 0] = forward_speed * t
        ref[:, 1] = radius * torch.sin(omega * t)
        ref[:, 2] = depth0 + radius * torch.cos(omega * t)
        ref[:, 6] = forward_speed
        ref[:, 7] = radius * omega * torch.cos(omega * t)
        ref[:, 8] = -radius * omega * torch.sin(omega * t)
        phi_amp = np.deg2rad(10);
        phi_freq = 0.2
        ref[:, 3] = phi_amp * torch.sin(phi_freq * t)
        return ref

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