function [traj_pos, traj_att, traj_vel, traj_omega, state_ref_ts] = trajectory_ref(cfg)
% 生成 BlueROV2 Heavy 参考轨迹
% 修改版：特技飞行轨迹 (Stunt Flight Trajectory)
% 特征：空间螺旋前进 + 大幅度三轴姿态复合机动
%
% 使用方式：
%   1) 作为函数调用：[traj_pos, ...] = trajectory_ref(cfg);
%   2) 作为脚本运行：直接运行 trajectory_ref.m，会自动使用默认参数并写入 workspace

% 如果作为脚本运行（无输入且无输出），使用默认参数
if nargin == 0 && nargout == 0
    cfg = struct;
    % 特技轨迹默认参数
    cfg.radius_start = 1.5;    % 螺旋滚筒半径 [m]
    cfg.radius_end   = 1.5;    % 保持半径
    cfg.omega        = 0.4;    % 轨迹盘旋频率 [rad/s]
    cfg.forward_speed = 0.3;   % 前进速度 [m/s]
    cfg.depth0       = 2;      % 中心深度 [m]
    cfg.sink_rate    = 0;      % 深度漂移率
    cfg.T_sim        = 100;     % 仿真时长 [s]
    cfg.dt           = 0.01;   % 采样时间 [s]
    
    % 生成轨迹
    [traj_pos, traj_att, traj_vel, traj_omega, state_ref_ts] = trajectory_ref(cfg);
    
    % 写入 base workspace
    assignin('base', 'traj_pos', traj_pos);
    assignin('base', 'traj_att', traj_att);
    assignin('base', 'traj_vel', traj_vel);
    assignin('base', 'traj_omega', traj_omega);
    assignin('base', 'state_ref_ts', state_ref_ts);
    
    disp('特技参考轨迹(Stunt Trajectory)已生成并写入 base workspace。');
    return;
end

% 函数模式：处理输入参数
if nargin == 0 || isempty(cfg)
    cfg = struct;
end

cfg = local_fill_default(cfg, 'radius_start', 1.5);
cfg = local_fill_default(cfg, 'radius_end',   1.5);
cfg = local_fill_default(cfg, 'omega',        0.4);
cfg = local_fill_default(cfg, 'forward_speed', 0.3);
cfg = local_fill_default(cfg, 'depth0',       2);
cfg = local_fill_default(cfg, 'sink_rate',    0);
cfg = local_fill_default(cfg, 'T_sim',        60);
cfg = local_fill_default(cfg, 'dt',           0.01);

t = (0:cfg.dt:cfg.T_sim)';

%% === 1. 位置轨迹：空间螺旋滚筒 ===
% X轴匀速，Y/Z轴画圆
x = cfg.forward_speed * t;
% Y = A * sin(wt)
y = cfg.radius_start * sin(cfg.omega * t); 
% Z = depth0 + A * cos(wt) (围绕中心深度盘旋)
z = cfg.depth0 + cfg.radius_start * cos(cfg.omega * t);

% 线速度 (位置的一阶导数)
vx = cfg.forward_speed * ones(size(t));
vy = cfg.radius_start * cfg.omega * cos(cfg.omega * t);
vz = -cfg.radius_start * cfg.omega * sin(cfg.omega * t);

%% === 2. 姿态轨迹：大幅度复合特技 ===
% Roll: 左右剧烈摇摆 (+/- 60度, 频率 0.6 rad/s)
phi_amp = deg2rad(60); 
phi_freq = 0.6;
phi = phi_amp * sin(phi_freq * t);
dphi = phi_amp * phi_freq * cos(phi_freq * t); % Roll 速率

% Pitch: 周期性俯冲拉起 (+/- 45度, 频率 0.4 rad/s)
theta_amp = deg2rad(45);
theta_freq = 0.4;
theta = theta_amp * sin(theta_freq * t);
dtheta = theta_amp * theta_freq * cos(theta_freq * t); % Pitch 速率

% Yaw: 缓慢偏航摆动 (+/- 30度, 频率 0.2 rad/s)
psi_amp = deg2rad(30);
psi_freq = 0.2;
psi = psi_amp * sin(psi_freq * t);
dpsi = psi_amp * psi_freq * cos(psi_freq * t); % Yaw 速率

%% === 3. 角速度计算 (Body Rates p,q,r) ===
% 使用欧拉角微分方程的逆关系计算对应的机体角速度
% p = dphi - dpsi * sin(theta)
% q = dtheta * cos(phi) + dpsi * sin(phi) * cos(theta)
% r = -dtheta * sin(phi) + dpsi * cos(phi) * cos(theta)

p = dphi - dpsi .* sin(theta);
q = dtheta .* cos(phi) + dpsi .* sin(phi) .* cos(theta);
r = -dtheta .* sin(phi) + dpsi .* cos(phi) .* cos(theta);

%% === 4. 数据打包输出 ===
traj_pos   = timeseries([x, y, z], t, 'Name', 'traj_pos');
traj_att   = timeseries([phi, theta, psi], t, 'Name', 'traj_att'); % 保存欧拉角
traj_vel   = timeseries([vx, vy, vz], t, 'Name', 'traj_vel');
traj_omega = timeseries([p, q, r], t, 'Name', 'traj_omega');        % 保存机体角速度

% 汇总状态向量 [x y z phi theta psi u v w p q r]
% 注意：这里的速度 state_ref(7:9) 存的是地球系线速度 vx,vy,vz
% 控制器中如果需要机体系速度，会自行转换。此处保持与原逻辑一致。
state_ref = [x, y, z, phi, theta, psi, ...
             vx, vy, vz, p, q, r];
state_ref_ts = timeseries(state_ref, t, 'Name', 'state_ref');
end

function cfg = local_fill_default(cfg, fieldName, defaultValue)
if ~isfield(cfg, fieldName) || isempty(cfg.(fieldName))
    cfg.(fieldName) = defaultValue;
end
end
