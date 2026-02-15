import torch

class ParallelESO:
    def __init__(self, config, num_envs, device):
        self.cfg = config
        self.num_envs = num_envs
        self.device = device
        self.dt = config.dt

        # ESO 带宽设置
        w0 = torch.tensor([4.0, 4.0, 4.0, 8.0, 8.0, 8.0], device=device).repeat(num_envs, 1)
        
        self.beta = torch.zeros(num_envs, 4, 6, device=device)
        self.beta[:, 0, :] = 4 * w0
        self.beta[:, 1, :] = 6 * w0 ** 2
        self.beta[:, 2, :] = 4 * w0 ** 3
        self.beta[:, 3, :] = w0 ** 4

        self.z = torch.zeros(num_envs, 24, device=device)
        self.initialized = False

    def reset(self, state_meas):
        self.z.fill_(0.0)
        self.z[:, 0:6] = state_meas[:, 0:6]
        self.initialized = True

    def update(self, state_meas, last_thrust_cmd_newton):
        B = self.num_envs

        # 1. 计算控制加速度 b*u
        tau_body = torch.matmul(self.cfg.T_mat, last_thrust_cmd_newton.unsqueeze(-1)).squeeze(-1)

        # [修复] 质量矩阵计算：使用 m + M_added (正确物理逻辑)，并且字段名改为 M_added
        # M_total = M_rb + M_a
        M_diag = torch.cat([
            self.cfg.m.squeeze() + self.cfg.M_added[0:3],
            self.cfg.I[:, 0] + self.cfg.M_added[3],
            self.cfg.I[:, 1] + self.cfg.M_added[4],
            self.cfg.I[:, 2] + self.cfg.M_added[5]
        ]).view(B, 6)
        
        acc_body = tau_body / M_diag

        # 2. 转换到 Earth Frame
        phi, theta, psi = state_meas[:, 3], state_meas[:, 4], state_meas[:, 5]
        J_vel, J_ang = self._compute_jacobian(phi, theta, psi)

        acc_earth_lin = torch.bmm(J_vel, acc_body[:, 0:3].unsqueeze(-1)).squeeze(-1)
        acc_earth_ang = torch.bmm(J_ang, acc_body[:, 3:6].unsqueeze(-1)).squeeze(-1)
        bu_earth = torch.cat([acc_earth_lin, acc_earth_ang], dim=1)

        # 3. RK4 更新
        k1 = self._eso_dynamics(self.z, state_meas, bu_earth)
        k2 = self._eso_dynamics(self.z + 0.5 * self.dt * k1, state_meas, bu_earth)
        k3 = self._eso_dynamics(self.z + 0.5 * self.dt * k2, state_meas, bu_earth)
        k4 = self._eso_dynamics(self.z + self.dt * k3, state_meas, bu_earth)
        self.z = self.z + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        
        # [新增] 对 ESO 内部状态的角度进行 Wrap，防止内部状态发散
        self.z[:, 3:6] = (self.z[:, 3:6] + torch.pi) % (2 * torch.pi) - torch.pi

        # 4. 提取扰动
        z3_est_earth = self.z[:, 12:18]
        
        # acc_dist_body = J_inv * z3
        J_full = torch.zeros(B, 6, 6, device=self.device)
        J_full[:, 0:3, 0:3] = J_vel
        J_full[:, 3:6, 3:6] = J_ang
        
        acc_dist_body = torch.linalg.solve(J_full, z3_est_earth.unsqueeze(-1)).squeeze(-1)
        dist_force_body = acc_dist_body * M_diag
        
        return dist_force_body, self.z

    def _eso_dynamics(self, z_curr, y_meas, bu):
        z1, z2 = z_curr[:, 0:6], z_curr[:, 6:12]
        z3, z4 = z_curr[:, 12:18], z_curr[:, 18:24]
        
        e = z1 - y_meas
        # [关键] 角度误差解缠 (Wrap-around)
        e[:, 3:6] = (e[:, 3:6] + torch.pi) % (2 * torch.pi) - torch.pi
        
        dz1 = z2 - self.beta[:, 0, :] * e
        dz2 = z3 - self.beta[:, 1, :] * e + bu
        dz3 = z4 - self.beta[:, 2, :] * e
        dz4 =    - self.beta[:, 3, :] * e
        
        return torch.cat([dz1, dz2, dz3, dz4], dim=1)

    def _compute_jacobian(self, phi, theta, psi):
        B = phi.shape[0]
        cph, sph = torch.cos(phi), torch.sin(phi)
        # [修复] 奇异性保护
        cth = torch.cos(theta)
        cth = torch.where(torch.abs(cth) < 1e-6, 1e-6 * torch.sign(cth), cth)
        sth = torch.sin(theta)
        
        cps, sps = torch.cos(psi), torch.sin(psi)

        J_vel = torch.zeros(B, 3, 3, device=self.device)
        J_vel[:, 0, 0] = cps * cth
        J_vel[:, 0, 1] = -sps * cph + cps * sth * sph
        J_vel[:, 0, 2] = sps * sph + cps * cph * sth
        J_vel[:, 1, 0] = sps * cth
        J_vel[:, 1, 1] = cps * cph + sps * sth * sph
        J_vel[:, 1, 2] = -cps * sph + sps * cph * sth
        J_vel[:, 2, 0] = -sth
        J_vel[:, 2, 1] = cth * sph
        J_vel[:, 2, 2] = cth * cph

        J_ang = torch.zeros(B, 3, 3, device=self.device)
        # tan(theta) = sth / cth (protected)
        tt = sth / cth
        
        J_ang[:, 0, 0] = 1.0
        J_ang[:, 0, 1] = sph * tt
        J_ang[:, 0, 2] = cph * tt
        J_ang[:, 1, 1] = cph
        J_ang[:, 1, 2] = -sph
        J_ang[:, 2, 1] = sph / cth
        J_ang[:, 2, 2] = cph / cth
        
        return J_vel, J_ang