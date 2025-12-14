import torch


class ParallelESO:
    def __init__(self, config, num_envs, device):
        self.cfg = config
        self.num_envs = num_envs
        self.device = device
        self.dt = config.dt

        # === 1. ESO 带宽设置 ===
        # 适当降低一点带宽以减少噪声，因为我们将移除外部滤波
        w0_lin = 4.0
        w0_ang = 8.0
        w0 = torch.tensor([w0_lin, w0_lin, w0_lin, w0_ang, w0_ang, w0_ang], device=device).repeat(num_envs, 1)

        # === 2. 增益系数 (标准 ESO 配置) ===
        self.beta = torch.zeros(num_envs, 2, 6, device=device)
        self.beta[:, 0, :] = 2.0 * w0  # beta1
        self.beta[:, 1, :] = w0 ** 2  # beta2

        # === 3. 状态变量 ===
        self.z = torch.zeros(num_envs, 12, device=device)

        # === [关键修复] 预计算质量矩阵对角线，防止运行时广播错误 ===
        # 显式扩展维度，确保 Added Mass 被正确加上
        m_rigid = self.cfg.m  # [N, 1]
        m_added_lin = self.cfg.M_added[0:3].unsqueeze(0).repeat(num_envs, 1)  # [N, 3]

        # 刚体质量 + 附加质量
        self.M_total_lin = m_rigid + m_added_lin  # [N, 3]

        # 转动惯量 + 附加惯量
        I_rigid = self.cfg.I  # [N, 3]
        I_added = self.cfg.M_added[3:6].unsqueeze(0).repeat(num_envs, 1)  # [N, 3]
        self.M_total_ang = I_rigid + I_added  # [N, 3]

        # 拼接成完整的对角质量向量 [N, 6]
        self.M_diag = torch.cat([self.M_total_lin, self.M_total_ang], dim=1)

        print(f"DEBUG: ESO Z-axis Total Mass = {self.M_diag[0, 2].item():.4f} kg (Target ~29.07)")

    def reset(self, velocity_meas):
        self.z.fill_(0.0)
        self.z[:, 0:6] = velocity_meas

    def update(self, velocity_meas, last_thrust_cmd_newton, orientation_meas):
        # --- 1. 计算已知力 (Known Forces) ---
        # 推进器推力
        tau_thruster = torch.matmul(self.cfg.T_mat, last_thrust_cmd_newton.unsqueeze(-1)).squeeze(-1)

        # 标称流体阻尼
        nu = velocity_meas
        D_vec = (self.cfg.D_lin + self.cfg.D_quad * torch.abs(nu)) * nu

        # 标称重力/浮力
        phi, theta = orientation_meas[:, 0], orientation_meas[:, 1]
        sth, cth = torch.sin(theta), torch.cos(theta)
        sph, cph = torch.sin(phi), torch.cos(phi)
        W, B_buoy = self.cfg.W, self.cfg.B

        g_vec = torch.zeros_like(nu)
        W_val = W.view(-1)
        # 注意：这里假设 NED 坐标系，Z 向下。重力是向下的力，如果 W>B，net force 是正的
        # 但 BlueROVDynamics 里定义 g_vec 是 "Restoring Force"，即要把 ROV 拉回水平的力
        # 所以 dynamics 是: M*acc = tau - D - g
        g_vec[:, 0] = (W_val - B_buoy) * sth
        g_vec[:, 1] = -(W_val - B_buoy) * cth * sph
        g_vec[:, 2] = -(W_val - B_buoy) * cth * cph

        rB = self.cfg.rB
        g_vec[:, 3] = -(rB[:, 1] * B_buoy * cth * cph - rB[:, 2] * B_buoy * cth * sph)
        g_vec[:, 4] = -(rB[:, 2] * B_buoy * sth + rB[:, 0] * B_buoy * cth * cph)
        g_vec[:, 5] = -(rB[:, 0] * B_buoy * cth * sph - rB[:, 1] * B_buoy * sth)

        # --- 2. 计算模型预测加速度 ---
        # 使用预计算好的 M_diag，确保包含了 Added Mass
        force_known = tau_thruster - D_vec - g_vec
        acc_model_body = force_known / self.M_diag

        # --- 3. ESO 更新 ---
        z1_est_vel = self.z[:, 0:6]
        z2_est_dist_acc = self.z[:, 6:12]

        error = z1_est_vel - velocity_meas

        dz1 = z2_est_dist_acc + acc_model_body - self.beta[:, 0, :] * error
        dz2 = - self.beta[:, 1, :] * error

        self.z[:, 0:6] += dz1 * self.dt
        self.z[:, 6:12] += dz2 * self.dt

        # --- 4. 输出 ---
        # 扰动加速度 -> 扰动力
        est_dist_force_body = self.z[:, 6:12] * self.M_diag

        return est_dist_force_body, self.z