import torch

# === 全局工具函数 ===
def skew_symmetric(v):
    """
    输入: [B, 3] -> 输出: [B, 3, 3] 反对称矩阵
    """
    B = v.shape[0]
    S = torch.zeros(B, 3, 3, device=v.device)
    S[:, 0, 1] = -v[:, 2]; S[:, 0, 2] =  v[:, 1]
    S[:, 1, 0] =  v[:, 2]; S[:, 1, 2] = -v[:, 0]
    S[:, 2, 0] = -v[:, 1]; S[:, 2, 1] =  v[:, 0]
    return S

def eul2rotm(phi, theta, psi):
    """
    输入: [B] -> 输出: [B, 3, 3] 旋转矩阵 R_b^e
    """
    cph, sph = torch.cos(phi), torch.sin(phi)
    cth, sth = torch.cos(theta), torch.sin(theta)
    cps, sps = torch.cos(psi), torch.sin(psi)
    B = phi.shape[0]
    R = torch.zeros(B, 3, 3, device=phi.device)
    R[:, 0, 0] = cps * cth
    R[:, 0, 1] = cps * sth * sph - sps * cph
    R[:, 0, 2] = cps * sth * cph + sps * sph
    R[:, 1, 0] = sps * cth
    R[:, 1, 1] = sps * sth * sph + cps * cph
    R[:, 1, 2] = sps * sth * cph - cps * sph
    R[:, 2, 0] = -sth
    R[:, 2, 1] = cth * sph
    R[:, 2, 2] = cth * cph
    return R

class ROVConfig:
    def __init__(self, num_envs=1, device='cuda'):
        self.device = device
        self.num_envs = num_envs
        self.dt = 0.01

        # === 物理参数 ===
        self.base_m = 14.5
        # 质量随机化
        self.m = self.base_m * (0.9 + 0.2 * torch.rand(num_envs, 1, device=device))

        self.g = 9.81
        self.W = self.m * self.g
        self.B = 143.0 

        # 惯性张量 (对角)
        self.Ixx = 0.25; self.Iyy = 0.35; self.Izz = 0.45
        self.I = torch.tensor([self.Ixx, self.Iyy, self.Izz], device=device).repeat(num_envs, 1)

        # [修复] 恢复 rB 定义 (浮心坐标，相对于原点)
        self.rB = torch.tensor([0.0, 0.0, -0.008], device=device).repeat(num_envs, 1)
        self.rG = torch.zeros(num_envs, 3, device=device)

        # [修复] 附加质量 (正值)，对应 M_total = M_rb + M_added
        self.M_added = torch.tensor([5.5, 12.7, 14.57, 0.12, 0.12, 0.12], device=device)

        # 线性阻尼 (正值系数，计算时加负号)
        self.D_lin = torch.tensor([4.03, 6.22, 5.18, 0.07, 0.07, 0.07], device=device)
        self.D_lin = self.D_lin * (0.8 + 0.4 * torch.rand(num_envs, 6, device=device))

        # 二次阻尼
        self.D_quad = torch.tensor([18.18, 21.66, 36.99, 1.55, 1.55, 1.55], device=device)

        # === 推进器 ===
        self.prop_max = 50.0
        self.prop_min = -40.0
        self.prop_tc = 0.15

        self.T_mat = self._compute_thrust_config().to(device)
        self.T_pinv = torch.linalg.pinv(self.T_mat).to(device)
        
        # 传感器噪声
        self.sens_noise_sigma = torch.tensor([
            0.1, 0.1, 0.02, 
            0.008, 0.008, 0.017,
            0.03, 0.03, 0.03,
            0.002, 0.002, 0.002
        ], device=device).unsqueeze(0)
        self.sens_gyro_bias = torch.randn(num_envs, 3, device=device) * 0.005
        
        # RL Scale
        self.rl_scale = torch.tensor([20., 20., 40., 10., 10., 10.], device=device)

    def _compute_thrust_config(self):
        pos = torch.tensor([
            [ 0.15,  0.15,   0], [ 0.15, -0.15,   0], [-0.15,  0.15,   0], [-0.15, -0.15,   0],
            [ 0.11,  0.22,   0.09], [-0.11,  0.22,   0.09], [ 0.11, -0.22,   0.09], [-0.11, -0.22,   0.09]
        ], device=self.device)
        dirs = torch.tensor([
            [ 0.7071, -0.7071,  0], [ 0.7071,  0.7071,  0], [-0.7071, -0.7071,  0], [-0.7071,  0.7071,  0],
            [ 0,       0,      -1], [ 0,       0,      -1], [ 0,       0,      -1], [ 0,       0,      -1]
        ], device=self.device)
        T = torch.zeros(6, 8, device=self.device)
        T[0:3, :] = dirs.T
        T[3:6, :] = torch.cross(pos, dirs, dim=1).T
        return T