%% BlueROV2 Heavy 6-DOF 仿真初始化 (参数统一版)
clear; clc;

%% === 1. 物理与水动力参数 (Standard Fossen Notation) ===
% 部分参数域随机化

base_m = 14.5;
params.m = base_m * (0.9 + 0.2 * rand()); % 质量浮动 +/- 10%

params.W   = params.m * 9.81;
params.B   = 143; 
params.rho = 1000;

% 惯性张量
params.Ixx = 0.25; params.Iyy = 0.35; params.Izz = 0.45;

base_rG = [0; 0; 0];
params.rG = base_rG + [0.01; 0.01; 0.005] .* randn(3,1); % 重心偏移

params.rB  = [0; 0; -0.008];

% 附加质量 (Added Mass)
params.Xu_dot = -5.5;   params.Yv_dot = -12.7;  params.Zw_dot = -14.57;
params.Kp_dot = -0.12;  params.Mq_dot = -0.12;  params.Nr_dot = -0.12;

% 线性阻尼 (Linear Damping) 浮动值


params.Xu = -4.03* (0.8 + 0.4 * rand()); params.Yv = -6.22* (0.8 + 0.4 * rand()); params.Zw = -5.18* (0.8 + 0.4 * rand());
params.Kp = -0.07* (0.9 + 0.2 * rand()); params.Mq = -0.07* (0.9 + 0.2 * rand()); params.Nr = -0.07* (0.9 + 0.2 * rand());

% 二次阻尼 (Quadratic Damping)
params.Xuu = -18.18; params.Yvv = -21.66; params.Zww = -36.99;
params.Kpp = -1.55;  params.Mqq = -1.55;  params.Nrr = -1.55;

%% === 2. 推进系统参数 (Propulsion) ===
% 统一前缀: prop_
params.prop_max = 50;  % [N] 最大正推力
params.prop_min = -40; % [N] 最大反推力
params.prop_rate_limit = 120; % [N/s] 物理推力变化率限制
params.prop_time_constant = 0.15; % [s] 电机一阶滞后时间常数

% 推进器几何布局
thruster_pos = [
     0.15,  0.15,   0;  0.15, -0.15,   0;
    -0.15,  0.15,   0; -0.15, -0.15,   0;
     0.11,  0.22,   0.09; -0.11,  0.22,   0.09;
     0.11, -0.22,   0.09; -0.11, -0.22,   0.09
];
% 矢量方向 (修正后的 V 型布局)
thruster_dir = [
     0.7071, -0.7071,  0;  0.7071,  0.7071,  0;
    -0.7071, -0.7071,  0; -0.7071,  0.7071,  0;
     0,       0,      -1;  0,       0,      -1;
     0,       0,      -1;  0,       0,      -1
];
% 归一化并计算配置矩阵
for i=1:8, thruster_dir(i,:) = thruster_dir(i,:)/norm(thruster_dir(i,:)); end
T_mat = zeros(6, 8);
for i = 1:8
    T_mat(1:3, i) = thruster_dir(i, :)';
    T_mat(4:6, i) = cross(thruster_pos(i, :)', thruster_dir(i, :)');
end

params.prop_T_matrix = T_mat;       % 推力配置矩阵
params.prop_T_pinv   = pinv(T_mat); % 伪逆分配矩阵

%% === 3. 传感器参数 (Sensors) ===
% 统一前缀: sens_
% 噪声标准差 Sigma
params.sens_sigma_pos   = [0.1; 0.1; 0.02];       % [m] USBL/Depth
params.sens_sigma_att   = deg2rad([0.5; 0.5; 1.0]); % [rad] IMU
params.sens_sigma_vel   = [0.03; 0.03; 0.03];     % [m/s] DVL
params.sens_sigma_omega = deg2rad([0.1; 0.1; 0.1]); % [rad/s] Gyro
% 汇总噪声向量 (12x1)
params.sens_noise_sigma = [params.sens_sigma_pos; params.sens_sigma_att; 
                           params.sens_sigma_vel; params.sens_sigma_omega];

% 陀螺仪随机零偏 (Bias)
bias_sigma_deg = 0.5; 
params.sens_gyro_bias = randn(3, 1) * deg2rad(bias_sigma_deg);
disp(['本次陀螺仪零偏 (x,y,z) [rad/s]: ', num2str(params.sens_gyro_bias')]);

%% === 4. 控制器参数 (Controllers) ===

% --- PID 控制器 (Prefix: pid_) ---
% 位置环
params.pid_Kp_pos = [30; 30; 50];    
params.pid_Ki_pos = [0.5; 0.5; 2.0];  
params.pid_Kd_pos = [10; 10; 25];    
params.pid_lim_int_pos = [5; 5; 10]; % 积分限幅

% 姿态环 (几何/欧拉共用)
params.pid_Kp_att = [60; 60; 60];    
params.pid_Ki_att = [3.0; 3.0; 3.0]; 
params.pid_Kd_att = [15; 15; 15];    
params.pid_lim_int_att = [5; 5; 5];

% --- SMC 控制器 ---
% 滑模面参数 Lambda (权重 e_dot vs e)
params.smc_Lambda = [1.5; 1.5; 1.5; 3.0; 3.0; 3.0]; 
% 积分增益 Ki
params.smc_Ki     = [0.1; 0.1; 0.2; 0.2; 0.2; 0.15];
% 切换增益 K_gain (鲁棒项幅度)
params.smc_K_gain = [20; 20; 40; 20; 20; 20]; 
% 边界层厚度 Phi (平滑度)
params.smc_Phi    = [0.5; 0.5; 0.5; 0.5; 0.5; 0.5];
% 积分总限幅
params.smc_lim_int = 10;

%% 5.洋流扰动参数
% 1. 设定洋流强度范围 (m/s)
% 0.1 m/s ~ 0.2 节 (微风)
% 0.5 m/s ~ 1.0 节 (强流，对 ROV 挑战很大)
current_min_speed = 0.1;
current_max_speed = 0.6; 

% 随机生成流速大小
current_speed = current_min_speed + (current_max_speed - current_min_speed) * rand();

% 2. 设定洋流方向 (随机 3D 方向)
% 偏航角 (0-360 deg)
current_yaw   = 2*pi * rand(); 
% 俯仰角 (通常洋流主要是水平的，但也可能有垂直分量)
current_pitch = deg2rad(10) * (2*rand() - 1); % +/- 10度倾角

% 转换为地球坐标系下的速度向量 [vx; vy; vz]
v_current_earth = current_speed * [
    cos(current_pitch) * cos(current_yaw);
    cos(current_pitch) * sin(current_yaw);
    -sin(current_pitch)            % NED坐标系，向上为负
];

params.prop_current_vel_earth = v_current_earth;

disp(['洋流环境生成: 速度 ' num2str(current_speed, '%.3f') ' m/s, ' ...
      '方向(Yaw) ' num2str(rad2deg(current_yaw), '%.1f') ' deg']);

% 静态波浪扰动偏移 (模拟持续的外部力矩，如缆绳拉力)
% 随机生成一个微小的常值力/力矩干扰
params.prop_wave_disturbance = [
    0.5 * randn(3,1);   % 力 (N)
    0.1 * randn(3,1)    % 力矩 (Nm)
];
%% 5.ESO参数
w0_pos = 4.0;  % 位置轴带宽 (x, y, z)
w0_att = 8.0;  % 姿态轴带宽 (phi, theta, psi)

params.eso_Gain_w0 = [
    w0_pos; w0_pos; w0_pos;
    w0_att; w0_att; w0_att
];

%% === 仿真配置 ===
params.dt    = 0.01;
params.T_sim = 100;

% 初始化状态
[traj_pos, ~, ~, ~, ~] = trajectory_ref(); 
start_pos = traj_pos.Data(1, :)'; % 拿到 t=0 的 [x, y, z]

% 2. 将 ROV 初始位置设为参考轨迹的起点
initial_state = zeros(12, 1);
initial_state(1:3) = start_pos; % 强制对齐位置
assignin('base', 'initial_state', initial_state);
assignin('base', 'params', params);
% 生成轨迹
trajectory_ref;