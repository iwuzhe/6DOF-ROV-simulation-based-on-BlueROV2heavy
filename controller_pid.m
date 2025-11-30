function tau = controller_pid(state_ref, state_act, params)
% 6-DOF 几何 PID 控制器 (参数统一版)

state_ref = state_ref(:);
state_act = state_act(:);

%% 1. 状态解析
p = state_act(1:3);
v = state_act(7:9);
R = eul2rotm_custom(state_act(4), state_act(5), state_act(6));
w = state_act(10:12);

p_d = state_ref(1:3);
v_d_earth = state_ref(7:9);
w_d = state_ref(10:12);
R_d = eul2rotm_custom(state_ref(4), state_ref(5), state_ref(6));

%% 2. 误差计算
% 位置误差 (Body Frame)
e_pos = R' * (p_d - p);
v_d_body = R' * v_d_earth;
e_vel = v_d_body - v;

% 姿态误差 (Geometric)
R_tilde = R_d' * R;
e_R_matrix = R_tilde - R_tilde';
e_att = 0.5 * vee_map(e_R_matrix);
e_omega = w - R' * R_d * w_d; 

%% 3. 积分项 (带抗饱和)
persistent int_e_pos int_e_att
if isempty(int_e_pos), int_e_pos = zeros(3,1); end
if isempty(int_e_att), int_e_att = zeros(3,1); end

int_e_pos = int_e_pos + e_pos * params.dt;
int_e_att = int_e_att + e_att * params.dt;

% [UPDATED] 使用统一命名的限幅参数
lim_p = params.pid_lim_int_pos;
lim_a = params.pid_lim_int_att;
int_e_pos = max(min(int_e_pos, lim_p), -lim_p);
int_e_att = max(min(int_e_att, lim_a), -lim_a);

%% 4. 控制律 (读取 pid_ 参数)
% [UPDATED] 读取 PID 增益
Kp_p = diag(params.pid_Kp_pos); Kd_p = diag(params.pid_Kd_pos); Ki_p = diag(params.pid_Ki_pos);
Kp_r = diag(params.pid_Kp_att); Kd_r = diag(params.pid_Kd_att); Ki_r = diag(params.pid_Ki_att);

% 位置力
F_body = Kp_p * e_pos + Kd_p * e_vel + Ki_p * int_e_pos;
% 姿态力矩
M_body = -Kp_r * e_att - Kd_r * e_omega - Ki_r * int_e_att;

%% 5. 重力补偿
f_g = R' * [0; 0; params.W];
f_b = R' * [0; 0; -params.B];
t_g = cross(params.rG, f_g) + cross(params.rB, f_b);

tau = [F_body - (f_g + f_b); M_body - t_g];
tau = tau(:);
end

% ... (辅助函数 eul2rotm_custom, vee_map 保持不变) ...
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