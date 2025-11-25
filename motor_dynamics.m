function thrust = motor_dynamics(thrust_cmd, thrust_prev, params)
% 修正版：一阶惯性环节 (First-Order Lag) + 限幅
% 模拟真实的电机加减速物理延迟

thrust_cmd = thrust_cmd(:);
thrust_prev = thrust_prev(:);

% 1. 定义时间常数 (Time Constant)
% T200 推进器的典型响应时间约为 0.1s 到 0.2s
% 如果 params 中没有定义，默认为 0.1s
if isfield(params, 'motor_tc')
    Tc = params.motor_tc;
else
    Tc = 0.1; 
end

% 2. 一阶滞后离散化 (Low-pass Filter)
% alpha = dt / (Tc + dt) 或简化为 dt / Tc
alpha = params.dt / Tc;

% 限制 alpha 在 0-1 之间，防止不稳定
alpha = max(min(alpha, 1), 0);

% 计算当前步的真实推力
thrust = thrust_prev + alpha * (thrust_cmd - thrust_prev);

% 3. 物理限幅 (最大/最小推力)
thrust = max(min(thrust, params.T_max), params.T_min);

% (可选) 仍然保留速率限制，模拟电机最大扭矩限制
% 如果需要双重限制，可以把之前的 Rate Limiter 加在 thrust_cmd 上
% 但通常一阶滞后已经包含了这个物理特性。

end