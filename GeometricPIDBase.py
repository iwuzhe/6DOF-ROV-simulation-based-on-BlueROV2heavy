import torch
from ROVConfig import eul2rotm


class GeometricPIDBase:
    def __init__(self, config):
        self.cfg = config
        d = config.device
        self.dt = config.dt

        # === PID 参数 (高增益抗扰动版) ===

        # 1. 位置环 [x, y, z]
        self.Kp_p = torch.diag(torch.tensor([40., 40., 80.], device=d))
        self.Kd_p = torch.diag(torch.tensor([15., 15., 35.], device=d))
        self.Ki_p = torch.diag(torch.tensor([0.5, 0.5, 2.0], device=d))

        # 2. 姿态环 [roll, pitch, yaw]
        # Kp 保持强劲，Kd 终于可以加大了！
        self.Kp_r = torch.diag(torch.tensor([30., 30., 30.], device=d))

        # [关键修复] 加大阻尼抑制物理震荡 (之前只能设15是因为噪声，现在有滤波了)
        self.Kd_r = torch.diag(torch.tensor([15., 15., 15.], device=d))

        # 稍微给点积分，消除静差
        self.Ki_r = torch.diag(torch.tensor([5.0, 5.0, 5.0], device=d))

        # 3. 积分限幅
        self.int_lim_p = torch.tensor([10.0, 10.0, 20.0], device=d)
        self.int_lim_r = torch.tensor([10.0, 10.0, 10.0], device=d)

        # 积分状态
        self.int_e_pos = torch.zeros(config.num_envs, 3, device=d)
        self.int_e_att = torch.zeros(config.num_envs, 3, device=d)

    def reset(self):
        self.int_e_pos.fill_(0.0)
        self.int_e_att.fill_(0.0)

    def compute_force(self, state, state_ref):
        # 1. 状态解析
        p_act = state[:, 0:3]
        v_act = state[:, 6:9]  # Body Frame
        w_act = state[:, 9:12]  # Body Frame
        R_act = eul2rotm(state[:, 3], state[:, 4], state[:, 5])

        p_des = state_ref[:, 0:3]
        v_ref_earth = state_ref[:, 6:9]  # Earth Frame
        w_ref_body = state_ref[:, 9:12]  # Body Frame
        R_des = eul2rotm(state_ref[:, 3], state_ref[:, 4], state_ref[:, 5])

        # 2. 误差计算

        # A. 位置误差
        e_pos_earth = p_des - p_act
        e_pos_body = torch.bmm(R_act.transpose(1, 2), e_pos_earth.unsqueeze(-1)).squeeze(-1)

        # B. 速度误差
        v_ref_body = torch.bmm(R_act.transpose(1, 2), v_ref_earth.unsqueeze(-1)).squeeze(-1)
        e_vel_body = v_ref_body - v_act

        # C. 姿态误差
        R_tilde = torch.bmm(R_des.transpose(1, 2), R_act)
        R_diff = R_tilde - R_tilde.transpose(1, 2)
        e_att_body = 0.5 * self._vee_map(R_diff)

        # D. 角速度误差
        R_map = torch.bmm(R_act.transpose(1, 2), R_des)
        w_ref_mapped = torch.bmm(R_map, w_ref_body.unsqueeze(-1)).squeeze(-1)
        e_omega = w_ref_mapped - w_act

        # 3. 积分更新
        self.int_e_pos += e_pos_body * self.dt
        self.int_e_att += e_att_body * self.dt
        self.int_e_pos = torch.max(torch.min(self.int_e_pos, self.int_lim_p), -self.int_lim_p)
        self.int_e_att = torch.max(torch.min(self.int_e_att, self.int_lim_r), -self.int_lim_r)

        # 4. 反馈力计算
        F_fb = (torch.matmul(self.Kp_p, e_pos_body.unsqueeze(-1)) +
                torch.matmul(self.Kd_p, e_vel_body.unsqueeze(-1)) +
                torch.matmul(self.Ki_p, self.int_e_pos.unsqueeze(-1))).squeeze(-1)

        M_fb = (-torch.matmul(self.Kp_r, e_att_body.unsqueeze(-1)) +
                torch.matmul(self.Kd_r, e_omega.unsqueeze(-1)) -
                torch.matmul(self.Ki_r, self.int_e_att.unsqueeze(-1))).squeeze(-1)

        # 5. 重力补偿
        g_vec = torch.zeros_like(state[:, 0:6])
        phi, theta = state[:, 3], state[:, 4]
        sth, cth = torch.sin(theta), torch.cos(theta)
        sph, cph = torch.sin(phi), torch.cos(phi)
        W, B_buoy = self.cfg.W, self.cfg.B

        g_vec[:, 0] = (W - B_buoy).squeeze() * sth
        g_vec[:, 1] = -(W - B_buoy).squeeze() * cth * sph
        g_vec[:, 2] = -(W - B_buoy).squeeze() * cth * cph

        rB = self.cfg.rB
        g_vec[:, 3] = -(rB[:, 1] * B_buoy * cth * cph - rB[:, 2] * B_buoy * cth * sph)
        g_vec[:, 4] = -(rB[:, 2] * B_buoy * sth + rB[:, 0] * B_buoy * cth * cph)
        g_vec[:, 5] = -(rB[:, 0] * B_buoy * cth * sph - rB[:, 1] * B_buoy * sth)

        tau_ff = g_vec

        # 总输出
        tau_out = torch.cat([F_fb, M_fb], dim=1) + tau_ff
        return tau_out

    def _vee_map(self, S):
        return torch.stack([-S[:, 1, 2], S[:, 0, 2], -S[:, 0, 1]], dim=1)