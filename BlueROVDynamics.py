import torch
from ROVConfig import ROVConfig, eul2rotm, skew_symmetric

class BlueROVDynamics:
    def __init__(self, config: ROVConfig):
        self.cfg = config
        self.device = config.device

    def dynamics_step(self, state, thrust_cmd_pwm, ocean_current_force=None):
        eta = state[:, 0:6]
        nu = state[:, 6:12]
        u, v, w = nu[:, 0], nu[:, 1], nu[:, 2]
        p, q, r = nu[:, 3], nu[:, 4], nu[:, 5]
        phi, theta, psi = eta[:, 3], eta[:, 4], eta[:, 5]
        B = state.shape[0]

        # --- 1. 质量矩阵 (M = M_rb + M_a) ---
        m_total = self.cfg.m.squeeze(1)
        M00 = m_total + self.cfg.M_added[0]
        M11 = m_total + self.cfg.M_added[1]
        M22 = m_total + self.cfg.M_added[2]
        M33 = self.cfg.I[:, 0] + self.cfg.M_added[3]
        M44 = self.cfg.I[:, 1] + self.cfg.M_added[4]
        M55 = self.cfg.I[:, 2] + self.cfg.M_added[5]
        
        M_vals = torch.stack([M00, M11, M22, M33, M44, M55], dim=1)
        M = torch.diag_embed(M_vals)

        # --- 2. 科里奥利矩阵 ---
        # Rigid Body Terms
        C_force = torch.zeros(B, 6, device=self.device)
        a1, a2, a3 = M00, M11, M22
        a4, a5, a6 = M33, M44, M55
        
        C_force[:, 0] = a3*w*q - a2*v*r
        C_force[:, 1] = a1*u*r - a3*w*p
        C_force[:, 2] = a2*v*p - a1*u*q
        C_force[:, 3] = (a6-a5)*q*r + (a2-a3)*v*w 
        C_force[:, 4] = (a4-a6)*r*p + (a3-a1)*w*u
        C_force[:, 5] = (a5-a4)*p*q + (a1-a2)*u*v

        # --- 3. 阻尼与恢复力 ---
        D_vec = (self.cfg.D_lin + self.cfg.D_quad * torch.abs(nu)) * nu
        
        g_vec = torch.zeros(B, 6, device=self.device)
        sth, cth = torch.sin(theta), torch.cos(theta)
        sph, cph = torch.sin(phi), torch.cos(phi)
        W, B_buoy = self.cfg.W, self.cfg.B
        
        g_vec[:, 0] = (W - B_buoy).squeeze() * sth
        g_vec[:, 1] = -(W - B_buoy).squeeze() * cth * sph
        g_vec[:, 2] = -(W - B_buoy).squeeze() * cth * cph
        
        # [修复] 使用 config 中的 rB (浮心坐标)
        rB = self.cfg.rB
        g_vec[:, 3] = -(rB[:,1]*B_buoy*cth*cph - rB[:,2]*B_buoy*cth*sph)
        g_vec[:, 4] = -(rB[:,2]*B_buoy*sth    + rB[:,0]*B_buoy*cth*cph)
        g_vec[:, 5] = -(rB[:,0]*B_buoy*cth*sph - rB[:,1]*B_buoy*sth)

        # --- 4. 求解 ---
        tau_thruster = (self.cfg.T_mat @ thrust_cmd_pwm.unsqueeze(-1)).squeeze(-1)
        total_rhs = tau_thruster - C_force - D_vec - g_vec
        if ocean_current_force is not None:
            total_rhs += ocean_current_force
            
        nu_dot = torch.linalg.solve(M, total_rhs.unsqueeze(-1)).squeeze(-1)

        # --- 5. 运动学 (带数值保护) ---
        # 防止 +/- 90度 除零错误
        cth_safe = torch.where(torch.abs(cth) < 1e-6, 1e-6 * torch.sign(cth), cth)
        tt_safe = sth / cth_safe # tan(theta)
        
        pos_dot = torch.bmm(eul2rotm(phi, theta, psi), nu[:, 0:3].unsqueeze(-1)).squeeze(-1)
        
        att_dot = torch.zeros_like(pos_dot)
        att_dot[:, 0] = p + (q*sph + r*cph) * tt_safe
        att_dot[:, 1] = q*cph - r*sph
        att_dot[:, 2] = (q*sph + r*cph) / cth_safe

        # 积分
        next_eta = eta + torch.cat([pos_dot, att_dot], dim=1) * self.cfg.dt
        next_nu = nu + nu_dot * self.cfg.dt
        
        # [修复] 姿态角缠绕处理 (Attitude Wrapping)
        # 将角度限制在 [-pi, pi]
        next_eta[:, 3:6] = (next_eta[:, 3:6] + torch.pi) % (2 * torch.pi) - torch.pi
        
        return torch.cat([next_eta, next_nu], dim=1)