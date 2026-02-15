function tau = controller_smc2(state_ref, state_act, params)
% 6-DOF 四元数反馈 SMC 控制器 (参数分离版)
% 
% 特点：
% 1. 核心算法不变：四元数误差解算 + 积分滑模面 + 模型补偿。
% 2. 参数分离：所有 Lambda, K, Ki, Phi 均从 params 读取。

state_ref = state_ref(:);
state_act = state_act(:);

%% 1. 状态解析
eta = state_act(1:6);   % [x y z phi theta psi]
nu  = state_act(7:12);  % [u v w p q r]
phi = eta(4); theta = eta(5); psi = eta(6);

eta_ref = state_ref(1:6);
nu_ref  = state_ref(7:12);
phi_d = eta_ref(4); theta_d = eta_ref(5); psi_d = eta_ref(6);

% 旋转矩阵 (Earth -> Body)
R_e2b = eul2rotm_custom(phi, theta, psi);

%% 2. 提取控制参数 (从 params 读取)
% 对应 init 文件中的 smc_ 前缀参数
Lambda = diag(params.smc_Lambda);
Ki     = diag(params.smc_Ki);
K_gain = params.smc_K;       % 切换增益
Phi    = params.smc_Phi;     % 边界层厚度
lim_int = params.smc_Int_limit; % 积分限幅

%% 3. 误差计算 (四元数核心)

% A. 位置误差 (Earth -> Body)
e_pos_earth = eta_ref(1:3) - eta(1:3);
e_pos_body = R_e2b' * e_pos_earth;

% B. 姿态误差 (四元数)
q_des = eul2quat_custom(phi_d, theta_d, psi_d);  % 4x1 或 1x4 任意
q_act = eul2quat_custom(phi, theta, psi);

% 统一为列向量并归一化（列向量形式）
q_des = q_des(:) / max(norm(q_des(:)), eps);
q_act = q_act(:) / max(norm(q_act(:)), eps);

% 计算共轭（列向量）
q_act_conj = [q_act(1); -q_act(2:4)];    % 4x1

% 如果 quatmultiply_custom 返回行向量或不确定，强制转列
q_err = quatmultiply_custom(q_des, q_act_conj);
q_err = q_err(:);  % 强制为 4x1 列向量

% 提取向量部分（列向量）
e_att_body = sign(q_err(1)) * q_err(2:4); 

% C. 速度误差
% 线速度 (需转到 Body)
v_ref_earth = nu_ref(1:3);
v_ref_body = R_e2b' * v_ref_earth;
e_vel_lin = v_ref_body - nu(1:3);

% 角速度 (直接在 Body)
e_vel_ang = nu_ref(4:6) - nu(4:6);

% 组合误差
e_pose = [e_pos_body; e_att_body];
e_vel  = [e_vel_lin; e_vel_ang];

%% 4. 积分滑模面设计
persistent int_e
if isempty(int_e), int_e = zeros(6,1); end

% 积分更新
int_e = int_e + e_pose * params.dt;

% 积分抗饱和 (Anti-windup)
int_e = max(min(int_e, lim_int), -lim_int);

% 滑模面 s = de + Lambda * e + Ki * int_e
s = e_vel + Lambda * e_pose + Ki * int_e;

%% 5. 模型补偿 (Feedforward)
% 重力补偿
W = params.W; B = params.B;
rG = params.rG; rB = params.rB;
sth=sin(theta); cth=cos(theta); sph=sin(phi); cph=cos(phi);

g_vec = zeros(6,1);
g_vec(1) = (W - B) * sth;
g_vec(2) = -(W - B) * cth * sph;
g_vec(3) = -(W - B) * cth * cph;
g_vec(4) = (rG(2)*W - rB(2)*B)*cth*cph - (rG(3)*W - rB(3)*B)*cth*sph;
g_vec(5) = -(rG(3)*W - rB(3)*B)*sth - (rG(1)*W - rB(1)*B)*cth*cph;
g_vec(6) = (rG(1)*W - rB(1)*B)*cth*sph + (rG(2)*W - rB(2)*B)*sth;

% 线性阻尼补偿
D_lin = -diag([params.Xu; params.Yv; params.Zw; params.Kp; params.Mq; params.Nr]);
tau_drag = D_lin * nu;

%% 6. 控制输出
% tau = g - D*nu + K * tanh(s/Phi)
tau = g_vec - tau_drag + K_gain .* tanh(s ./ Phi);

% 维度整形
tau = tau(:);

end

%% === 辅助函数 ===
function R = eul2rotm_custom(phi, theta, psi)
cph = cos(phi); sph = sin(phi);
cth = cos(theta); sth = sin(theta);
cps = cos(psi); sps = sin(psi);
R = [cps*cth, cps*sth*sph-sps*cph, cps*sth*cph+sps*sph;
     sps*cth, sps*sth*sph+cps*cph, sps*sth*cph-cps*sph;
     -sth,    cth*sph,             cth*cph];
end

function q = eul2quat_custom(phi, theta, psi)
c1 = cos(psi/2); s1 = sin(psi/2);
c2 = cos(theta/2); s2 = sin(theta/2);
c3 = cos(phi/2); s3 = sin(phi/2);
w = c1*c2*c3 + s1*s2*s3;
x = c1*c2*s3 - s1*s2*c3;
y = c1*s2*c3 + s1*c2*s3;
z = s1*c2*c3 - c1*s2*s3;
q = [w; x; y; z];
end

function q_out = quatmultiply_custom(q, r)
q0 = q(1); q1 = q(2); q2 = q(3); q3 = q(4);
r0 = r(1); r1 = r(2); r2 = r(3); r3 = r(4);
q_out = [
    q0*r0 - q1*r1 - q2*r2 - q3*r3;
    q0*r1 + q1*r0 + q2*r3 - q3*r2;
    q0*r2 - q1*r3 + q2*r0 + q3*r1;
    q0*r3 + q1*r2 - q2*r1 + q3*r0
];
end