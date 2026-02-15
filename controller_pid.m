function tau = controller_pid(state_ref, state_act, params)
% 6-DOF 几何 PID 控制器 (Geometric PID on SE(3))
% 
% 核心原理：
%   利用旋转矩阵 R 计算流形上的姿态误差 e_R。
%   控制律：tau = -Kp*e_R - Kd*e_w - Ki*int_e + g(x)
%   特点：结构简单，无奇点，适合大角度机动。

state_ref = state_ref(:);
state_act = state_act(:);

%% 1. 状态解析
% 实际状态
p = state_act(1:3);
v = state_act(7:9); % 机体线速度
R = eul2rotm_custom(state_act(4), state_act(5), state_act(6)); % Body->Earth
w = state_act(10:12); % 机体角速度

% 期望状态
p_d = state_ref(1:3);
v_d_earth = state_ref(7:9); % 期望线速度 (Earth Frame)
w_d = state_ref(10:12);     % 期望角速度 (Body Frame)
R_d = eul2rotm_custom(state_ref(4), state_ref(5), state_ref(6));

%% 2. 几何误差计算 (Geometric Errors)

% --- A. 位置误差 (转换到机体坐标系) ---
% e_p = R' * (p_d - p)
e_pos = R' * (p_d - p);

% --- B. 速度误差 (机体坐标系) ---
v_d_body = R' * v_d_earth;
e_vel = v_d_body - v;

% --- C. 姿态误差 (基于旋转矩阵) ---
% 定义相对旋转误差矩阵: R_tilde = R_d^T * R
% 这一步计算了"实际姿态"相对于"期望姿态"偏了多少
R_tilde = R_d' * R;

% 从 R_tilde 中提取误差向量 e_R
% e_R = 0.5 * vee(R_tilde - R_tilde')
% 物理意义：e_R 的方向是旋转轴，模长是 sin(theta)
e_R_matrix = R_tilde - R_tilde';
e_att = 0.5 * vee_map(e_R_matrix);

% --- D. 角速度误差 ---
% 将期望角速度转换到当前机体坐标系下对比
% e_w = w - R^T * R_d * w_d
e_omega = w - R' * R_d * w_d; 
% 注意：这里的符号定义 e_w = w - w_d (近似)，PID公式里用 -Kd*e_w

%% 3. 积分项 (Integral Action)
persistent int_e_pos int_e_att
if isempty(int_e_pos), int_e_pos = zeros(3,1); end
if isempty(int_e_att), int_e_att = zeros(3,1); end

dt = params.dt;

% 累加误差
int_e_pos = int_e_pos + e_pos * dt;
int_e_att = int_e_att + e_att * dt;

% 积分抗饱和 (读取 params 设置)
lim_p = params.Int_limit_pos;
lim_a = params.Int_limit_att;
int_e_pos = max(min(int_e_pos, lim_p), -lim_p);
int_e_att = max(min(int_e_att, lim_a), -lim_a);

%% 4. 获取 PID 参数 (参数分离)
Kp_p = diag(params.Kp_pos); Kd_p = diag(params.Kd_pos); Ki_p = diag(params.Ki_pos);
Kp_r = diag(params.Kp_att); Kd_r = diag(params.Kd_att); Ki_r = diag(params.Ki_att);

%% 5. 控制律计算 (Control Law)

% --- A. 位置控制力 (Body Frame) ---
% F = Kp*e_p + Kd*e_v + Ki*int_e
F_body = Kp_p * e_pos + Kd_p * e_vel + Ki_p * int_e_pos;

% --- B. 姿态控制力矩 (Body Frame) ---
% M = -Kp*e_R - Kd*e_w - Ki*int_e
% 注意负号：因为 e_att 和 e_omega 定义为 (Actual - Desired) 的方向
M_body = -Kp_r * e_att - Kd_r * e_omega - Ki_r * int_e_att;

%% 6. 重力补偿 (Gravity Compensation)
% 即使是 PID，也必须补偿重力，否则会有稳态下垂
W = params.W; B = params.B;
rG = params.rG; rB = params.rB;

% 重力/浮力在机体坐标系下的分量
f_g = R' * [0; 0; W];
f_b = R' * [0; 0; -B];

% 恢复力矩
% M_g = rG x f_g + rB x f_b
t_g = cross(rG, f_g) + cross(rB, f_b);

% 补偿向量 (我们要输出力去抵消它，所以取反)
F_comp = -(f_g + f_b);
M_comp = -t_g;

%% 7. 总输出
tau = [F_body + F_comp; M_body + M_comp];

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

function v = vee_map(S)
    % 将反对称矩阵映射回向量
    v = [-S(2,3); S(1,3); -S(1,2)];
end