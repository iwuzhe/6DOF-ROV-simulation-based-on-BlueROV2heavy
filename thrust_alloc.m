function thrust_cmd = thrust_alloc(tau, params)
% 推力分配：tau (6x1) -> thrust_cmd (8x1)

tau = tau(:);
thrust_cmd = params.T_pinv * tau;

thrust_cmd = max(min(thrust_cmd, params.T_max), params.T_min);
end

