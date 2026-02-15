function tau = controller_geometricpid(state_ref, state_act, params)
% 6-DOF 增强型几何控制器 (参数分离版)
% 逻辑：完全依赖 params 输入，不再内部硬编码增益

state_ref = state_ref(:);
state_act = state_act(:);

% === 1. 状态解析 ===
p_act = state_act(1:3);
v_act = state_act(7:9);
R_act = eul2rotm_custom(state_act(4), state_act(5), state_act(6));
w_act = state_act(10:12);

p_des = state_ref(1:3);
v_ref_earth = state_ref(7:9); 
w_ref_body  = state_ref(10:12); 
R_des = eul2rotm_custom(state_ref(4), state_ref(5), state_ref(6));

% === 2. 提取参数 (从 params 读取) ===
% 位置环增益
Kp_p = diag(params.Kp_pos);
Kd_p = diag(params.Kd_pos);
Ki_p = diag(params.Ki_pos);

% 姿态环增益
Kp_r = diag(params.Kp_att);
Kd_r = diag(params.Kd_att);
Ki_r = diag(params.Ki_att);

% 积分限幅
lim_p_int = params.Int_limit_pos;
lim_a_int = params.Int_limit_att;

% 物理参数
m = params.m;
J_mat = diag([params.Ixx, params.Iyy, params.Izz]);

% === 3. 角加速度前馈 (wd_dot) ===
persistent last_w_ref
if isempty(last_w_ref), last_w_ref = zeros(3,1); end
wd_dot = (w_ref_body - last_w_ref) / params.dt;
last_w_ref = w_ref_body;
wd_dot = max(min(wd_dot, 10), -10); % 简单限幅

% === 4. 误差计算 ===
% 位置与速度误差 (Body Frame)
e_pos_body = R_act' * (p_des - p_act);
v_ref_body = R_act' * v_ref_earth;
e_vel_body = v_ref_body - v_act;

% 姿态误差 (Geometric Error)
R_err_mat = R_des' * R_act - R_act' * R_des;
e_att_body = 0.5 * vee_map(R_err_mat);
e_omega = w_ref_body - w_act;

% === 5. 积分项 ===
persistent int_e_pos int_e_att
if isempty(int_e_pos), int_e_pos = zeros(3,1); end
if isempty(int_e_att), int_e_att = zeros(3,1); end

int_e_pos = int_e_pos + e_pos_body * params.dt;
int_e_att = int_e_att + e_att_body * params.dt;

% 抗饱和 (使用 params 中的限幅值)
int_e_pos = max(min(int_e_pos, lim_p_int), -lim_p_int);
int_e_att = max(min(int_e_att, lim_a_int), -lim_a_int);

% === 6. 力与力矩计算 ===
% 反馈项
F_fb = Kp_p * e_pos_body + Kd_p * e_vel_body + Ki_p * int_e_pos;
M_fb = -Kp_r * e_att_body + Kd_r * e_omega - Ki_r * int_e_att;

% 前馈项 (动力学补偿)
g_earth = [0; 0; params.W - params.B]; 
f_g_body = R_act' * g_earth;

f_W_body = R_act' * [0;0;params.W];
f_B_body = R_act' * [0;0;-params.B];
M_restoring = cross(params.rG, f_W_body) + cross(params.rB, f_B_body);

M_gyro = cross(w_act, J_mat * w_act);
M_inertial = J_mat * wd_dot;

D_lin_force = -diag([params.Xu; params.Yv; params.Zw]) * v_act;
D_lin_torque = -diag([params.Kp; params.Mq; params.Nr]) * w_act;

% 总输出
F_total = F_fb - f_g_body - D_lin_force; 
M_total = M_fb - M_restoring + M_gyro + M_inertial - D_lin_torque;

tau = [F_total; M_total];
tau = tau(:);
end

function R = eul2rotm_custom(phi, theta, psi)
cph = cos(phi); sph = sin(phi);
cth = cos(theta); sth = sin(theta);
cps = cos(psi); sps = sin(psi);
R = [cps*cth, cps*sth*sph-sps*cph, cps*sth*cph+sps*sph;
     sps*cth, sps*sth*sph+cps*cph, sps*sth*cph-cps*sph;
     -sth,    cth*sph,             cth*cph];
end

function v = vee_map(S)
v = [-S(2,3); S(1,3); -S(1,2)];
end