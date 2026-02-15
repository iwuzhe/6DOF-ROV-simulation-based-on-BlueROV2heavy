import torch
from ROVConfig import eul2rotm, skew_symmetric

class GeometricPIDBase:
    def __init__(self, config):
        self.cfg = config
        d = config.device
        # PID 参数
        self.Kp_p = torch.diag(torch.tensor([40., 40., 80.], device=d))
        self.Kd_p = torch.diag(torch.tensor([15., 15., 35.], device=d))
        self.Kp_r = torch.diag(torch.tensor([60., 60., 60.], device=d))
        self.Kd_r = torch.diag(torch.tensor([15., 15., 15.], device=d))

    def compute_force(self, state, state_ref):
        p_act = state[:, 0:3]
        v_act = state[:, 6:9]
        w_act = state[:, 9:12]
        R_act = eul2rotm(state[:, 3], state[:, 4], state[:, 5])

        p_des = state_ref[:, 0:3]
        v_ref_earth = state_ref[:, 6:9]
        w_ref_body = state_ref[:, 9:12]
        R_des = eul2rotm(state_ref[:, 3], state_ref[:, 4], state_ref[:, 5])

        # 1. Error Terms
        e_pos_earth = p_des - p_act
        e_pos_body = torch.bmm(R_act.transpose(1, 2), e_pos_earth.unsqueeze(-1)).squeeze(-1)
        
        v_ref_body = torch.bmm(R_act.transpose(1, 2), v_ref_earth.unsqueeze(-1)).squeeze(-1)
        e_vel_body = v_ref_body - v_act
        
        R_err_mat = torch.bmm(R_des.transpose(1, 2), R_act) - torch.bmm(R_act.transpose(1, 2), R_des)
        e_att_body = 0.5 * self._vee_map(R_err_mat)
        e_omega = w_ref_body - w_act

        # 2. Feedback Force
        F_fb = (torch.matmul(self.Kp_p, e_pos_body.unsqueeze(-1)) +
                torch.matmul(self.Kd_p, e_vel_body.unsqueeze(-1))).squeeze(-1)

        M_fb = (-torch.matmul(self.Kp_r, e_att_body.unsqueeze(-1)) +
                torch.matmul(self.Kd_r, e_omega.unsqueeze(-1))).squeeze(-1)

        # 3. [关键修复] 重力补偿 (Feedforward)
        # 根据 M*a = tau - g，平衡状态需 tau = g
        # 且 g_vec 已经包含重力与浮力的合力（在Body系下）
        g_vec = torch.zeros_like(state[:, 0:6])
        phi, theta = state[:, 3], state[:, 4]
        sth, cth = torch.sin(theta), torch.cos(theta)
        sph, cph = torch.sin(phi), torch.cos(phi)
        W, B_buoy = self.cfg.W, self.cfg.B
        
        # Force Part
        g_vec[:, 0] = (W - B_buoy).squeeze() * sth
        g_vec[:, 1] = -(W - B_buoy).squeeze() * cth * sph
        g_vec[:, 2] = -(W - B_buoy).squeeze() * cth * cph
        
        # Moment Part (使用 config 中的 rB)
        rB = self.cfg.rB
        g_vec[:, 3] = -(rB[:,1]*B_buoy*cth*cph - rB[:,2]*B_buoy*cth*sph)
        g_vec[:, 4] = -(rB[:,2]*B_buoy*sth    + rB[:,0]*B_buoy*cth*cph)
        g_vec[:, 5] = -(rB[:,0]*B_buoy*cth*sph - rB[:,1]*B_buoy*sth)
        
        tau_ff = g_vec
        
        # 4. Total Output
        tau_out = torch.cat([F_fb, M_fb], dim=1) + tau_ff
        
        return tau_out

    def _vee_map(self, S):
        return torch.stack([-S[:, 1, 2], S[:, 0, 2], -S[:, 0, 1]], dim=1)