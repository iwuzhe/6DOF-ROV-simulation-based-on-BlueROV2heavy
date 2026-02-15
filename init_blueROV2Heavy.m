%% BlueROV2 Heavy 6-DOF 仿真初始化
% 作用：在 MATLAB base workspace 中生成 params 结构体与参考轨迹
clear; clc;
clear controller_pid;
clear controller_smc;
%% 1. 物理与水动力参数
params.m   = 14.5;
params.W   = params.m * 9.81;
params.B   = 143;
params.rho = 1000;

params.Ixx = 0.25;
params.Iyy = 0.35;
params.Izz = 0.45;

params.rG = [0; 0; 0];
params.rB = [0; 0; -0.008];

% 附加质量
params.Xu_dot = -5.5;
params.Yv_dot = -12.7;
params.Zw_dot = -14.57;
params.Kp_dot = -0.12;
params.Mq_dot = -0.12;
params.Nr_dot = -0.12;

% 线性阻尼
params.Xu = -4.03;
params.Yv = -6.22;
params.Zw = -5.18;
params.Kp = -0.07;
params.Mq = -0.07;
params.Nr = -0.07;

% 二次阻尼
params.Xuu = -18.18;
params.Yvv = -21.66;
params.Zww = -36.99;
params.Kpp = -1.55;
params.Mqq = -1.55;
params.Nrr = -1.55;

%% 2.噪声参数
%传感器噪声
% 1. 位置噪声 (x, y, z)
% x,y 通常由 USBL 或视觉里程计提供，噪声较大 (例如 0.1m)
% z 通常由压力传感器(深度计)提供，精度很高 (例如 0.02m)
sigma_pos = [0.1; 0.1; 0.02]; 

% 2. 姿态噪声 (phi, theta, psi)
sigma_att_deg = [0.5; 0.5; 1.0]; % 偏航角(psi)磁力计干扰大，噪声稍大
sigma_att = deg2rad(sigma_att_deg);

% 3. 线速度噪声 (u, v, w)
% DVL (多普勒测速仪) 测量值，通常精度在 0.01 - 0.05 m/s
sigma_vel = [0.03; 0.03; 0.03]; 

% 4. 角速度噪声 (p, q, r)
% 陀螺仪原始数据噪声，取决于器件 (例如 0.1 deg/s)
sigma_omega_deg = [0.1; 0.1; 0.1];
sigma_omega = deg2rad(sigma_omega_deg);

params.sensor_noise_sigma = [sigma_pos; sigma_att; sigma_vel; sigma_omega];

disp('传感器噪声参数 (Sigma) 已配置。');

bias_sigma_deg = 0.5; % 单位: deg/s
bias_sigma_rad = deg2rad(bias_sigma_deg);
params.sensor_gyro_bias = randn(3, 1) * bias_sigma_rad;
disp(['本次陀螺仪零偏 (x,y,z) [rad/s]: ', num2str(params.sensor_gyro_bias')]);
%% 3. 推进器几何
thruster_pos = [
     0.15,  0.15,   0;
     0.15, -0.15,   0;
    -0.15,  0.15,   0;
    -0.15, -0.15,   0;
     0.11,  0.22,   0.09;
    -0.11,  0.22,   0.09;
     0.11, -0.22,   0.09;
    -0.11, -0.22,   0.09
];

thruster_dir = [
     0.7071, -0.7071,  0;  % T1: 指向左前方 (产生 Yaw 力矩)
     0.7071,  0.7071,  0;  % T2: 指向右前方
    -0.7071, -0.7071,  0;  % T3: 指向左后方
    -0.7071,  0.7071,  0;  % T4: 指向右后方
     0,       0,      -1;  % T5: 垂向 (注意这里通常是 -1 向上推，或者 1 向下推，根据你的推力定义)
     0,       0,      -1;  % T6
     0,       0,      -1;  % T7
     0,       0,      -1   % T8
];

for i = 1:8
    thruster_dir(i, :) = thruster_dir(i, :) / norm(thruster_dir(i, :));
end

T = zeros(6, 8);
for i = 1:8
    T(1:3, i) = thruster_dir(i, :)';
    T(4:6, i) = cross(thruster_pos(i, :)', thruster_dir(i, :)');
end
params.T = T;
params.T_pinv = pinv(T);

params.T_max = 50;
params.T_min = -40;
params.thrust_rate = 120;  % N/s

%% 4. 仿真参数
params.dt    = 0.01;
params.T_sim = 100;

initial_state = zeros(12, 1);
assignin('base', 'initial_state', initial_state);

% --- 位置环 ---
% --- 位置环 (Position) ---
% Z轴(3)需要较大积分项来消除重力估算误差
params.Kp_pos = [30; 30; 50];    
params.Ki_pos = [0.5; 0.5; 2.0];  
params.Kd_pos = [10; 10; 25];    
params.Int_limit_pos = [5; 5; 10];

% --- 姿态环 (Attitude - Geometric) ---
% 几何误差 e_R 的最大模长只有 1 (sin theta)，而欧拉角可以很大
% 所以几何 PID 的 Kp 通常需要给得比欧拉 PID 大
% Pitch(2) 和 Roll(1) 需要强积分来对抗浮力矩
params.Kp_att = [60; 60; 60];    
params.Ki_att = [3.0; 3.0; 3.0]; 
params.Kd_att = [15; 15; 15];    
params.Int_limit_att = [5; 5; 5];

params.smc_Lambda = [1.5; 1.5; 1.5; 3.0; 3.0; 3.0]; 

% 积分增益 Ki (消除稳态误差)
params.smc_Ki     = [0.1; 0.1; 0.2; 0.2; 0.2; 0.15];

% 切换增益 K (决定抗干扰能力，需大于扰动上界)
params.smc_K      = [20; 20; 40; 20; 20; 20]; 

% 边界层厚度 Phi (平滑抖振，值越大越平滑但精度越低)
params.smc_Phi    = [0.5; 0.5; 0.5; 0.5; 0.5; 0.5];

% 积分限幅
params.smc_Int_limit = 10;
%% 5. 写入 base workspace
assignin('base', 'params', params);

disp('BlueROV2 Heavy 参数已写入 base workspace。');
disp('提示：运行 trajectory_ref.m 可生成参考轨迹。');
trajectory_ref;
out = sim("BlueROV2_Heavy_6DOF.slx");
