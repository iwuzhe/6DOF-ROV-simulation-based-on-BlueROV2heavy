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

        # State Init
        self.state = torch.zeros(num_envs, 12, device=device)
        self.last_thrust = torch.zeros(num_envs, 8, device=device)
        self.time = 0.0

        # ESO Data Holders
        self.est_dist_body = torch.zeros(num_envs, 6, device=device)
        self.est_state = torch.zeros(num_envs, 12, device=device)

    def reset(self):
        # 1. 获取参考轨迹起点
        state_ref_0 = self.get_reference_trajectory(0.0)

        # 2. 强制对齐状态
        self.state = state_ref_0.clone()
        
        # 3. 随机扰动 (仅在训练时建议开启，测试时可注释)
        if self.num_envs > 1:
            self.state[:, 3:6] += (torch.rand(self.num_envs, 3, device=self.device) - 0.5) * 0.2
            
        self.last_thrust.fill_(0.0)
        self.time = 0.0

        # 4. ESO 复位
        meas = self.sensor_model(self.state)
        self.eso.reset(meas[:, 0:6])

        return self.get_observation(state_ref_0)

    def step(self, action_rl_residual):
        # 1. 获取参考轨迹
        state_ref = self.get_reference_trajectory(self.time)

        # 2. 传感器与 ESO 更新
        obs_meas = self.sensor_model(self.state)
        dist_force_body, eso_internal_state = self.eso.update(obs_meas[:, 0:6], self.last_thrust)

        self.est_dist_body = dist_force_body
        self.est_state = torch.cat([eso_internal_state[:, 0:6], eso_internal_state[:, 6:12]], dim=1)

        # 3. 基准控制器 (PID)
        tau_pid = self.base_ctrl.compute_force(self.est_state, state_ref)

        # 4. 力融合 (Residual RL)
        tau_rl = action_rl_residual * self.cfg.rl_scale
        tau_total = tau_pid + tau_rl 

        # 5. 推力分配与执行
        thrust_cmd = torch.matmul(self.cfg.T_pinv, tau_total.unsqueeze(-1)).squeeze(-1)
        thrust_cmd = torch.clamp(thrust_cmd, self.cfg.prop_min, self.cfg.prop_max)

        actual_thrust = self.motor_dynamics(thrust_cmd, self.last_thrust)
        self.last_thrust = actual_thrust

        # 6. 物理环境步进
        real_env_force = self.calc_ocean_current(self.state)
        self.state = self.plant.dynamics_step(self.state, actual_thrust, real_env_force)
        self.time += self.cfg.dt

        # 7. 观测与奖励
        obs = self.get_observation(state_ref)

        error = self.state - state_ref
        reward = -torch.norm(error[:, 0:3], dim=1) - 0.5 * torch.norm(error[:, 3:6], dim=1) \
                 - 0.05 * torch.norm(action_rl_residual, dim=1)

        done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        return obs, reward, done, {}

    def get_observation(self, state_ref):
        e_p = state_ref[:, 0:3] - self.est_state[:, 0:3]
        e_att = state_ref[:, 3:6] - self.est_state[:, 3:6]
        e_v = state_ref[:, 6:9] - self.est_state[:, 6:9]
        e_w = state_ref[:, 9:12] - self.est_state[:, 9:12]

        return torch.cat([
            e_p, e_att, e_v, e_w,
            self.est_dist_body,
            self.est_state
        ], dim=1)

    def calc_ocean_current(self, state):
        # 简单的常值洋流模拟
        v_c_earth = torch.tensor([0.5, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)

        phi, theta, psi = state[:, 3], state[:, 4], state[:, 5]
        R = eul2rotm(phi, theta, psi)

        # 转到机体系
        v_c_body = torch.bmm(R.transpose(1, 2), v_c_earth.unsqueeze(-1)).squeeze(-1)

        # 相对速度
        nu = state[:, 6:12]
        nu_r = nu.clone()
        nu_r[:, 0:3] -= v_c_body

        # 计算阻力差
        D_coef = self.cfg.D_quad 
        F_drag_real = (self.cfg.D_lin + D_coef * torch.abs(nu_r)) * nu_r
        F_drag_nom = (self.cfg.D_lin + D_coef * torch.abs(nu)) * nu

        tau_env = F_drag_nom - F_drag_real
        return tau_env

    def get_reference_trajectory(self, t):
        # 1. 统一转为 Tensor
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=self.device)
        
        # [关键修复] 如果 t 是标量(0维)，强制扩充为向量 [B]
        if t.ndim == 0:
            t = t.expand(self.num_envs)

        ref = torch.zeros(self.num_envs, 12, device=self.device)
        
        # 螺旋参数
        omega = 0.4
        radius_y = 1.0
        radius_z = 1.0
        vx_val = 0.2
        
        # --- A. 位置 (Pos) ---
        pos_x = vx_val * t
        pos_y = radius_y * torch.sin(omega * t)
        pos_z = 2.0 + radius_z * torch.cos(omega * t)
        
        ref[:, 0] = pos_x
        ref[:, 1] = pos_y
        ref[:, 2] = pos_z
        
        # --- B. 线速度 (Vel) ---
        vel_x = vx_val * torch.ones_like(t)
        vel_y = radius_y * omega * torch.cos(omega * t)
        vel_z = -radius_z * omega * torch.sin(omega * t)
        
        ref[:, 6] = vel_x
        ref[:, 7] = vel_y
        ref[:, 8] = vel_z
        
        # --- C. 姿态 (Attitude) - 切向跟随 ---
        v_horz_sq = vel_x**2 + vel_y**2
        v_horz = torch.sqrt(v_horz_sq)
        
        # Yaw: 跟随水平速度方向
        psi = torch.atan2(vel_y, vel_x)
        # Pitch: 跟随爬升角
        theta = torch.atan2(-vel_z, v_horz)
        # Roll: 强制水平
        phi = torch.zeros_like(t)
        
        ref[:, 3] = phi
        ref[:, 4] = theta
        ref[:, 5] = psi
        
        # --- D. [关键修复] 解析角速度 (Angular Velocity) ---
        acc_x = torch.zeros_like(t)
        acc_y = -radius_y * (omega**2) * torch.sin(omega * t)
        acc_z = -radius_z * (omega**2) * torch.cos(omega * t)
        
        # Yaw Rate
        dpsi = (vel_x * acc_y - vel_y * acc_x) / (v_horz_sq + 1e-6)
        
        # Pitch Rate
        u, v = -vel_z, v_horz
        du = -acc_z
        dv = (vel_x * acc_x + vel_y * acc_y) / (v_horz + 1e-6)
        dtheta = (v * du - u * dv) / (u**2 + v**2 + 1e-6)
        
        # Roll Rate
        dphi = torch.zeros_like(t)
        
        # 转换到机体坐标系
        sth, cth = torch.sin(theta), torch.cos(theta)
        sph, cph = torch.sin(phi), torch.cos(phi)
        
        p = dphi - dpsi * sth
        q = dtheta * cph + dpsi * sph * cth
        r = -dtheta * sph + dpsi * cph * cth
        
        ref[:, 9] = p
        ref[:, 10] = q
        ref[:, 11] = r
        
        return ref

    def sensor_model(self, state):
        noise = torch.randn_like(state) * self.cfg.sens_noise_sigma
        bias = torch.zeros_like(state)
        bias[:, 9:12] = self.cfg.sens_gyro_bias
        return state + noise + bias

    def motor_dynamics(self, cmd, prev):
        alpha = self.cfg.dt / self.cfg.prop_tc
        thrust = prev + alpha * (cmd - prev)
        noise = torch.randn_like(thrust) * 0.5
        return thrust + noise