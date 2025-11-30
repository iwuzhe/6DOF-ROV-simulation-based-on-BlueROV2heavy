function tau_env = calc_ocean_current_force(state_act, params)
% CALC_OCEAN_CURRENT_FORCE 计算洋流对ROV产生的6维环境干扰力
% 
% 原理：
%   根据 Fossen 模型，水动阻力取决于相对速度：nu_r = nu - nu_current
%   本函数计算扰动项：tau_env = D(nu)*nu - D(nu_r)*nu_r
%   将此力加到 ROV 动力学方程中，等效于让 ROV 在流动的海洋中运动。
%
% 输入：
%   state_act : 12x1 实际状态 [x y z phi theta psi u v w p q r]
%   params    : 参数结构体 (需包含 prop_current_vel_earth 等)
%
% 输出：
%   tau_env   : 6x1 干扰力向量 (Body Frame)

    % 1. 解析状态
    eta = state_act(1:6);
    nu  = state_act(7:12);
    
    phi = eta(4); theta = eta(5); psi = eta(6);
    u = nu(1); v = nu(2); w = nu(3);
    p = nu(4); q = nu(5); r = nu(6);
    
    % 2. 获取洋流参数 (地球坐标系 NED)
    % 如果 params 中没有定义，默认为 0
    if isfield(params, 'prop_current_vel_earth')
        v_c_earth = params.prop_current_vel_earth; % [vx_c; vy_c; vz_c]
    else
        v_c_earth = [0; 0; 0];
    end
    
    % 3. 将洋流速度转换到机体坐标系 (Body Frame)
    % 旋转矩阵 R_eb (Earth to Body) = R_be'
    R_be = eul2rotm_custom(phi, theta, psi);
    v_c_body = R_be' * v_c_earth;
    
    % 4. 计算相对速度 (Relative Velocity)
    % 洋流主要影响线速度，通常假设流场是无旋的，所以角速度相对量不变
    nu_r = nu; 
    nu_r(1:3) = nu(1:3) - v_c_body; 
    
    % 5. 读取阻力系数 (支持你已经做好的随机化参数)
    Xu = params.Xu; Yv = params.Yv; Zw = params.Zw;
    Kp = params.Kp; Mq = params.Mq; Nr = params.Nr;
    
    Xuu = params.Xuu; Yvv = params.Yvv; Zww = params.Zww;
    Kpp = params.Kpp; Mqq = params.Mqq; Nrr = params.Nrr;
    
    % 6. 计算标称阻力 (Nominal Damping Force) - 假设静水
    % 对应 rov_plant 中原本的 -D*nu 项
    D_nom_vec = [
        (Xu + Xuu*abs(u)) * u;
        (Yv + Yvv*abs(v)) * v;
        (Zw + Zww*abs(w)) * w;
        (Kp + Kpp*abs(p)) * p;
        (Mq + Mqq*abs(q)) * q;
        (Nr + Nrr*abs(r)) * r
    ];
    
    % 7. 计算实际阻力 (Real Damping Force) - 基于相对速度
    % 使用 nu_r 计算
    u_r = nu_r(1); v_r = nu_r(2); w_r = nu_r(3);
    
    D_real_vec = [
        (Xu + Xuu*abs(u_r)) * u_r;
        (Yv + Yvv*abs(v_r)) * v_r;
        (Zw + Zww*abs(w_r)) * w_r;
        (Kp + Kpp*abs(p)) * p;  % 角运动通常受洋流影响较小，除非考虑非对称性
        (Mq + Mqq*abs(q)) * q;
        (Nr + Nrr*abs(r)) * r
    ];
    
    % 8. 计算扰动补偿力
    % 动力学方程目标： M*acc = tau_thrust - D_real - g
    % 当前 Plant方程： M*acc = tau_thrust - D_nom  - g + tau_env
    % 因此： tau_env = D_nom - D_real
    
    % 注意：阻力系数通常为负值 (如 Xu = -4.03)，因此阻力项本身是负的
    % 这里我们保持代数符号一致性：Force = Coefficient * Velocity
    
    tau_env = D_nom_vec - D_real_vec;
    
    % (可选) 增加一个缓慢变化的扰动成分（模拟波浪的一阶马尔可夫过程）
    % 如果 params 中有定义 wave_force
    if isfield(params, 'prop_wave_disturbance')
        tau_env = tau_env + params.prop_wave_disturbance;
    end

end

%% 辅助函数：欧拉角转旋转矩阵
function R = eul2rotm_custom(phi, theta, psi)
    cph = cos(phi); sph = sin(phi);
    cth = cos(theta); sth = sin(theta);
    cps = cos(psi); sps = sin(psi);
    R = [cps*cth, cps*sth*sph-sps*cph, cps*sth*cph+sps*sph;
         sps*cth, sps*sth*sph+cps*cph, sps*sth*cph-cps*sph;
         -sth,    cth*sph,             cth*cph];
end