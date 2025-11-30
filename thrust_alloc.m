function thrust_cmd = thrust_alloc(tau, params)
% 推力分配 (参数统一版)
% tau (6x1) -> thrust_cmd (8x1)

tau = tau(:);

% [UPDATED] 使用统一的推力参数
thrust_cmd = params.prop_T_pinv * tau;

% 饱和截断
thrust_cmd = max(min(thrust_cmd, params.prop_max), params.prop_min);

end