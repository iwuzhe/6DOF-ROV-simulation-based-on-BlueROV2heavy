function tau = controller_smc2(state_ref, state_act, params)
% 6-DOF 四元数 SMC 控制器 (参数统一版)

state_ref = state_ref(:);
state_act = state_act(:);

%% 1. 状态解析
eta = state_act(1:6); nu  = state_act(7:12);
phi = eta(4); theta = eta(5); psi = eta(6);

eta_ref = state_ref(1:6); nu_ref  = state_ref(7:12);
phi_d = eta_ref(4); theta_d = eta_ref(5); psi_d = eta_ref(6);
R_e2b = eul2rotm_custom(phi, theta, psi);

%% 2. 提取参数 (Updated Names)
% [UPDATED] 使用 smc_ 前缀参数
Lambda = diag(params.smc_Lambda);
Ki     = diag(params.smc_Ki);
K_gain = params.smc_K_gain;    % 切换增益
Phi    = params.smc_Phi;       % 边界层
lim_int = params.smc_lim_int;  % 积分限幅

%% 3. 误差计算 (省略部分与原逻辑一致，重点检查参数引用)
% ... (Position Error) ...
e_pos_earth = eta_ref(1:3) - eta(1:3);
e_pos_body = R_e2b' * e_pos_earth;

% ... (Quaternion Error) ...
q_des = eul2quat_custom(phi_d, theta_d, psi_d);
q_act = eul2quat_custom(phi, theta, psi);
q_des = q_des(:)/norm(q_des); q_act = q_act(:)/norm(q_act);
q_act_conj = [q_act(1); -q_act(2:4)];
q_err = quatmultiply_custom(q_des, q_act_conj);
e_att_body = sign(q_err(1)) * q_err(2:4);

% ... (Velocity Error) ...
v_ref_body = R_e2b' * nu_ref(1:3);
e_vel = [v_ref_body - nu(1:3); nu_ref(4:6) - nu(4:6)];
e_pose = [e_pos_body; e_att_body];

%% 4. 滑模面
persistent int_e
if isempty(int_e), int_e = zeros(6,1); end
int_e = int_e + e_pose * params.dt;
int_e = max(min(int_e, lim_int), -lim_int);

s = e_vel + Lambda * e_pose + Ki * int_e;

%% 5. 模型补偿
% 重力项 (Physics params 保持 Fossen 命名)
W = params.W; B = params.B; rG = params.rG; rB = params.rB;
% ... (g_vec 计算逻辑保持不变) ...
sth=sin(theta); cth=cos(theta); sph=sin(phi); cph=cos(phi);
g_vec = zeros(6,1);
g_vec(1) = (W - B) * sth;
g_vec(2) = -(W - B) * cth * sph;
g_vec(3) = -(W - B) * cth * cph;
g_vec(4) = (rG(2)*W - rB(2)*B)*cth*cph - (rG(3)*W - rB(3)*B)*cth*sph;
g_vec(5) = -(rG(3)*W - rB(3)*B)*sth - (rG(1)*W - rB(1)*B)*cth*cph;
g_vec(6) = (rG(1)*W - rB(1)*B)*cth*sph + (rG(2)*W - rB(2)*B)*sth;

% 阻尼项 (Physics params)
D_lin = -diag([params.Xu; params.Yv; params.Zw; params.Kp; params.Mq; params.Nr]);
tau_drag = D_lin * nu;

%% 6. 输出
tau = g_vec - tau_drag + K_gain .* tanh(s ./ Phi);
tau = tau(:);
end

% ... (辅助函数 eul2rotm, eul2quat, quatmultiply 保持不变) ...
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
    q = [c1*c2*c3 + s1*s2*s3; c1*c2*s3 - s1*s2*c3; c1*s2*c3 + s1*c2*s3; s1*c2*c3 - c1*s2*s3];
end

function q_out = quatmultiply_custom(q, r)
    q0=q(1); q1=q(2); q2=q(3); q3=q(4);
    r0=r(1); r1=r(2); r2=r(3); r3=r(4);
    q_out = [q0*r0 - q1*r1 - q2*r2 - q3*r3;
             q0*r1 + q1*r0 + q2*r3 - q3*r2;
             q0*r2 - q1*r3 + q2*r0 + q3*r1;
             q0*r3 + q1*r2 - q2*r1 + q3*r0];
end