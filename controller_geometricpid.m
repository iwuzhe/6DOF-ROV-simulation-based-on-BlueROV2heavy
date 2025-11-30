function tau = controller_geometricpid(state_ref, state_act, params)
% 6-DOF 增强型几何控制器 (参数统一版)
% 
% 核心逻辑：PID 反馈 + 动力学模型前馈 (Feedforward)
% 适用场景：高速机动、特技飞行、已知模型参数的高精度控制

state_ref = state_ref(:);
state_act = state_act(:);

%% 1. 状态解析
% 实际状态
p_act = state_act(1:3);
v_act = state_act(7:9);
R_act = eul2rotm_custom(state_act(4), state_act(5), state_act(6));
w_act = state_act(10:12);

% 期望状态
p_des = state_ref(1:3);
v_ref_earth = state_ref(7:9); 
w_ref_body  = state_ref(10:12); 
R_des = eul2rotm_custom(state_ref(4), state_ref(5), state_ref(6));

%% 2. 提取参数 (使用统一命名规范)
% [UPDATED] 使用 pid_ 前缀的增益参数
% 注意：几何PID通常可以与普通PID共用一套增益，或者需要调得更激进
Kp_p = diag(params.pid_Kp_pos);
Kd_p = diag(params.pid_Kd_pos);
Ki_p = diag(params.pid_Ki_pos);

Kp_r = diag(params.pid_Kp_att);
Kd_r = diag(params.pid_Kd_att);
Ki_r = diag(params.pid_Ki_att);

% [UPDATED] 积分限幅
lim_p_int = params.pid_lim_int_pos;
lim_a_int = params.pid_lim_int_att;

% 物理参数 (用于前馈)
m = params.m;
J_mat = diag([params.Ixx, params.Iyy, params.Izz]);

%% 3. 角加速度前馈 (wd_dot)
persistent last_w_ref
if isempty(last_w_ref), last_w_ref = zeros(3,1); end
% 简单差分计算期望角加速度
wd_dot = (w_ref_body - last_w_ref) / params.dt;
last_w_ref = w_ref_body;
% 简单限幅防止微分噪声
wd_dot = max(min(wd_dot, 20), -20); 

%% 4. 误差计算 (Geometric on SE(3))
% 位置与速度误差 (转到 Body Frame)
e_pos_body = R_act' * (p_des - p_act);
v_ref_body = R_act' * v_ref_earth;
e_vel_body = v_ref_body - v_act;

% 姿态误差 (基于旋转矩阵)
% R_err = R_des^T * R_act - R_act^T * R_des
R_err_mat = R_des' * R_act - R_act' * R_des;
e_att_body = 0.5 * vee_map(R_err_mat);

% 角速度误差
e_omega = w_ref_body - w_act; % 注意方向定义需与 PID 符号匹配

%% 5. 积分项
persistent int_e_pos int_e_att
if isempty(int_e_pos), int_e_pos = zeros(3,1); end
if isempty(int_e_att), int_e_att = zeros(3,1); end

int_e_pos = int_e_pos + e_pos_body * params.dt;
int_e_att = int_e_att + e_att_body * params.dt;

% 抗饱和
int_e_pos = max(min(int_e_pos, lim_p_int), -lim_p_int);
int_e_att = max(min(int_e_att, lim_a_int), -lim_a_int);

%% 6. 力与力矩计算

% --- A. 反馈项 (Feedback) ---
% F_fb = Kp*e + Kd*e_dot + Ki*int
F_fb = Kp_p * e_pos_body + Kd_p * e_vel_body + Ki_p * int_e_pos;
% M_fb = -Kp*e_R + Kd*e_w - Ki*int (注意姿态误差的符号定义)
M_fb = -Kp_r * e_att_body + Kd_r * e_omega - Ki_r * int_e_att;

% --- B. 前馈项 (Feedforward / Model Compensation) ---
% 1. 重力/浮力补偿 (Body Frame)
g_earth = [0; 0; params.W - params.B]; 
f_g_body = R_act' * g_earth; % 重浮力合力在机体系分量

f_W_body = R_act' * [0;0;params.W];
f_B_body = R_act' * [0;0;-params.B];
% 恢复力矩 (Restoring Torque)
M_restoring = cross(params.rG, f_W_body) + cross(params.rB, f_B_body);

% 2. 惯性力矩前馈 (Gyroscopic + Inertial)
M_gyro = cross(w_act, J_mat * w_act);
M_inertial = J_mat * wd_dot;

% 3. 水动力阻力前馈 (Drag Compensation)
% 这一步非常关键：直接根据当前速度计算预计阻力，并输出反向力抵消
D_lin_force = -diag([params.Xu; params.Yv; params.Zw]) * v_act;
D_lin_torque = -diag([params.Kp; params.Mq; params.Nr]) * w_act;

% 二次阻尼前馈 (可选，如果模型够准可以加，这里先加线性的)
% D_quad_force = ... 

%% 7. 总输出
% F_total = 反馈 - 重力 - 阻力 (注意：阻力本身是负的，所以减去它是为了产生正推力)
F_total = F_fb - f_g_body - D_lin_force; 

% M_total = 反馈 - 恢复力矩 + 陀螺 + 惯性 - 阻尼力矩
M_total = M_fb - M_restoring + M_gyro + M_inertial - D_lin_torque;

tau = [F_total; M_total];
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

function v = vee_map(S)
    v = [-S(2,3); S(1,3); -S(1,2)];
end