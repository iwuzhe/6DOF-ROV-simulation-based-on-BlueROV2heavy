function thrust = motor_dynamics(thrust_cmd, thrust_prev, params)
% 电机动力学
% 模拟一阶滞后 (First-Order Lag) 和限幅
thrust_cmd = thrust_cmd(:);
thrust_prev = thrust_prev(:);
Tc = params.prop_time_constant;
alpha = params.dt / Tc;
alpha = max(min(alpha, 1), 0);
%滞后更新
thrust = thrust_prev + alpha * (thrust_cmd - thrust_prev);

% 推力噪声
   % 1. 乘性噪声: 模拟电机效率波动
    efficiency_noise = 1 + 0.05 * randn(size(thrust_cmd)); 
    
    % 2. 加性噪声: 模拟水流紊流对螺旋桨的干扰
    turbulence = 0.5 * randn(size(thrust_cmd)); % 假设 0.5N 的抖动

thrust = thrust .* efficiency_noise + turbulence;
% 物理限幅 (prop_max / prop_min)
thrust = max(min(thrust, params.prop_max), params.prop_min);

end