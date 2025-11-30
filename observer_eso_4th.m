function [dist_force_body, estimated_state] = observer_eso_4th(state_meas, thrust_cmd, params)
% OBSERVER_ESO_4TH 4阶扩张状态观测器 (RK4)
%
% 动力学模型:
%   dot(z1) = z2 - beta1 * e
%   dot(z2) = z3 - beta2 * e + b*u (可选已知模型)
%   dot(z3) = z4 - beta3 * e
%   dot(z4) =    - beta4 * e
%   其中 e = z1 - y (测量误差)
%
% 输入:
%   state_meas : 6x1 传感器测量位姿 [x y z phi theta psi] (Earth Frame)
%   thrust_cmd : 8x1 电机推力指令 (用于计算已知加速度 b*u)
%   params     : 参数结构体 (需包含 eso_Gain_w0)
%
% 输出:
%   dist_force_body : 6x1 估计的合外力/干扰力 (转换回 Body Frame，可直接补偿)
%   estimated_state : 24x1 完整观测状态 [z1; z2; z3; z4] (用于调试/绘图)

    % 1. 初始化持久变量 (24x1 向量: 6DOF * 4States)
    persistent z
    if isempty(z)
        % 初始化 z1 为当前测量值，防止初始误差过大
        z = zeros(24, 1);
        z(1:6) = state_meas; 
    end

    dt = params.dt;
    
    % 2. 提取参数与增益
    % 使用带宽参数化法 (Bandwidth Parameterization)
    % beta1 = 4*w0, beta2 = 6*w0^2, beta3 = 4*w0^3, beta4 = w0^4
    if isfield(params, 'eso_Gain_w0')
        w0 = params.eso_Gain_w0; % 6x1 向量，每个轴的带宽
    else
        w0 = 5 * ones(6, 1); % 默认带宽
    end
    
    beta = zeros(4, 6);
    beta(1,:) = 4 .* w0';
    beta(2,:) = 6 .* w0'.^2;
    beta(3,:) = 4 .* w0'.^3;
    beta(4,:) = w0'.^4;

    % 计算机体推力
    tau_body = params.prop_T_matrix * thrust_cmd;
    
    % 转换到地球系加速度: a_earth = J * M^-1 * tau_body
    eta_meas = state_meas;
    phi=eta_meas(4); theta=eta_meas(5); psi=eta_meas(6);
    
    % 雅可比矩阵 J(eta)
    J_mat = [
        cos(psi)*cos(theta), -sin(psi)*cos(phi)+cos(psi)*sin(theta)*sin(phi), sin(psi)*sin(phi)+cos(psi)*cos(phi)*sin(theta);
        sin(psi)*cos(theta), cos(psi)*cos(phi)+sin(phi)*sin(theta)*sin(psi), -cos(psi)*sin(phi)+sin(theta)*sin(psi)*cos(phi);
        -sin(theta),          cos(theta)*sin(phi),                             cos(theta)*cos(phi)
    ];
    R_ang = [
        1, sin(phi)*tan(theta), cos(phi)*tan(theta);
        0, cos(phi),            -sin(phi);
        0, sin(phi)/cos(theta), cos(phi)/cos(theta)
    ];
    J_full = blkdiag(J_mat, R_ang);
    
    % 质量逆矩阵
    M_inv = diag(1 ./ [
        params.m - params.Xu_dot; params.m - params.Yv_dot; params.m - params.Zw_dot;
        params.Ixx - params.Kp_dot; params.Iyy - params.Mq_dot; params.Izz - params.Nr_dot
    ]);

    % 地球系控制加速度 (6x1)
    acc_ctrl_earth = J_full * (M_inv * tau_body);

    % 4. RK4 更新 (核心)
    % y = state_meas
    
    k1 = eso_dynamics(z,          state_meas, acc_ctrl_earth, beta);
    k2 = eso_dynamics(z + 0.5*dt*k1, state_meas, acc_ctrl_earth, beta);
    k3 = eso_dynamics(z + 0.5*dt*k2, state_meas, acc_ctrl_earth, beta);
    k4 = eso_dynamics(z + dt*k3,     state_meas, acc_ctrl_earth, beta);
    
    z_next = z + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4);
    z = z_next;
    
    % 5. 输出处理
    % z3 是地球系下的总扰动加速度 (Estimated Acceleration Disturbance)
    z3_est = z(13:18); % 索引 13-18 是第3阶状态
    
    % 将估计的加速度转换为机体坐标系下的力/力矩 (用于控制补偿)
    % F_dist_body = M * J^-1 * a_dist_earth
    % 注意：这里 J^-1 * z3 近似为机体系下的角加速度
    % 简便计算：直接反解
    
    acc_body_est = J_full \ z3_est; % 转回机体系加速度
    M_mat = diag(1./diag(M_inv));   % 还原质量矩阵
    
    dist_force_body = M_mat * acc_body_est;
    
    estimated_state = z(1:12);
end

%% 辅助函数：ESO 微分方程
function dz = eso_dynamics(z_curr, y_meas, bu, beta)
    % z_curr: 24x1 [z1; z2; z3; z4]
    % y_meas: 6x1
    % bu:     6x1 (Known Control Acceleration)
    % beta:   4x6 (Gains)
    
    % 1. 拆分状态
    z1 = z_curr(1:6);
    z2 = z_curr(7:12);
    z3 = z_curr(13:18);
    z4 = z_curr(19:24);
    
    % 2. 计算误差 e = z1 - y
    e = z1 - y_meas;
    
    % [关键] 角度误差解缠 (Wrap-around handling)
    % 针对欧拉角 (4,5,6 维)，处理 +/- pi 跳变
    for i = 4:6
        e(i) = atan2(sin(e(i)), cos(e(i))); % 映射回 [-pi, pi]
    end
    
    % 3. 状态方程
    % dot(z1) = z2 - beta1*e
    dz1 = z2 - beta(1,:)'.*e;
    
    % dot(z2) = z3 - beta2*e + b*u
    % 这里 z3 代表"总加速度"中的"扰动+未建模部分" (如果加了 bu)
    % 或者 z3 代表"总加速度" (如果不加 bu)
    % 为了控制补偿方便，这里加上 bu，这样 z3 输出的就是纯扰动加速度
    dz2 = z3 - beta(2,:)'.*e + bu;
    
    % dot(z3) = z4 - beta3*e
    dz3 = z4 - beta(3,:)'.*e;
    
    % dot(z4) =    - beta4*e
    dz4 =    - beta(4,:)'.*e;
    
    dz = [dz1; dz2; dz3; dz4];
end