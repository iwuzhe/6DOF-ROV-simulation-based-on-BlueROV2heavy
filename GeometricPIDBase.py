import torch
from ROVConfig import eul2rotm


class GeometricPIDBase:
    def __init__(self, config):
        self.cfg = config
        d = config.device
        self.dt = config.dt

        # === PID 参数 (严格对齐 MATLAB controller_pid.m) ===
        # 位置环 [x, y, z]
        self.Kp_p = torch.diag(torch.tensor([30., 30., 50.], device=d))  # MATLAB: [30; 30; 50]
        self.Kd_p = torch.diag(torch.tensor([10., 10., 25.], device=d))  # MATLAB: [10; 10; 25]
        self.Ki_p = torch.diag(torch.tensor([0.5, 0.5, 2.0], device=d))  # MATLAB: [0.5; 0.5; 2.0]

        # 姿态环 [roll, pitch, yaw]
        self.Kp_r = torch.diag(torch.tensor([60., 60., 60.], device=d))  # MATLAB: [60; 60; 60]
        self.Kd_r = torch.diag(torch.tensor([15., 15., 15.], device=d))  # MATLAB: [15; 15; 15]
        self.Ki_r = torch.diag(torch.tensor([3.0, 3.0, 3.0], device=d))  # MATLAB: [3.0; 3.0; 3.0]

        # 积分限幅
        self.int_lim_p = torch.tensor([5.0, 5.0, 10.0], device=d)
        self.int_lim_r = torch.tensor([5.0, 5.0, 5.0], device=d)

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

        # 2. 误差计算 (严格复刻 MATLAB 逻辑)

        # A. 位置误差 (转到机体系)
        # MATLAB: e_pos = R' * (p_d - p)
        e_pos_earth = p_des - p_act
        e_pos_body = torch.bmm(R_act.transpose(1, 2), e_pos_earth.unsqueeze(-1)).squeeze(-1)

        # B. 速度误差 (转到机体系)
        # MATLAB: v_d_body = R' * v_d_earth; e_vel = v_d_body - v
        v_ref_body = torch.bmm(R_act.transpose(1, 2), v_ref_earth.unsqueeze(-1)).squeeze(-1)
        e_vel_body = v_ref_body - v_act

        # C. 姿态误差 (几何误差)
        # MATLAB: R_tilde = R_d' * R; e_R = 0.5 * vee(R_tilde - R_tilde')
        # 注意: MATLAB 代码中 R_tilde = R_d' * R 是 "Desired to Actual" 的误差矩阵
        R_tilde = torch.bmm(R_des.transpose(1, 2), R_act)
        R_diff = R_tilde - R_tilde.transpose(1, 2)
        e_att_body = 0.5 * self._vee_map(R_diff)

        # D. 角速度误差 (关键点)
        # MATLAB: e_omega = w - R' * R_d * w_d
        # 含义: e_omega = Actual - Desired_Mapped_To_Actual
        # 我们的 PID 公式是 -Kd * (Actual - Desired) 或者 +Kd * (Desired - Actual)
        # 这里我们需要计算 (Desired - Actual) 以便使用 +Kd

        # 将参考角速度映射到当前机体系
        # w_ref_mapped = R_act^T * R_des * w_ref_body
        R_map = torch.bmm(R_act.transpose(1, 2), R_des)
        w_ref_mapped = torch.bmm(R_map, w_ref_body.unsqueeze(-1)).squeeze(-1)

        e_omega = w_ref_mapped - w_act  # Desired - Actual

        # 3. 积分更新
        self.int_e_pos += e_pos_body * self.dt
        self.int_e_att += e_att_body * self.dt
        # 抗饱和
        self.int_e_pos = torch.max(torch.min(self.int_e_pos, self.int_lim_p), -self.int_lim_p)
        self.int_e_att = torch.max(torch.min(self.int_e_att, self.int_lim_r), -self.int_lim_r)

        # 4. 反馈力计算
        # F = Kp * e_p + Kd * e_v + Ki * int_e
        F_fb = (torch.matmul(self.Kp_p, e_pos_body.unsqueeze(-1)) +
                torch.matmul(self.Kd_p, e_vel_body.unsqueeze(-1)) +
                torch.matmul(self.Ki_p, self.int_e_pos.unsqueeze(-1))).squeeze(-1)

        # M = -Kp * e_R + Kd * e_w - Ki * int_e
        # 注意符号: e_att_body 是 (Act - Des) 的方向，所以用 -Kp
        # e_omega 是 (Des - Act) 的方向，所以用 +Kd
        M_fb = (-torch.matmul(self.Kp_r, e_att_body.unsqueeze(-1)) +
                torch.matmul(self.Kd_r, e_omega.unsqueeze(-1)) -
                torch.matmul(self.Ki_r, self.int_e_att.unsqueeze(-1))).squeeze(-1)

        # 5. 重力补偿 (仅此一项，无其他前馈)
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

        # M*a = tau - g_vec -> tau = M*a + g_vec.
        # 为了抵消重力，我们需要输出 +g_vec
        tau_ff = g_vec

        # 总输出
        tau_out = torch.cat([F_fb, M_fb], dim=1) + tau_ff
        return tau_out

    def _vee_map(self, S):
        return torch.stack([-S[:, 1, 2], S[:, 0, 2], -S[:, 0, 1]], dim=1)