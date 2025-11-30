function state_meas = sensor_model(state_true, params)
    % 传感器噪声模型 (参数统一版)
    
    % [UPDATED] 1. 高斯白噪声 (读取 sens_noise_sigma)
    noise = randn(size(state_true)) .* params.sens_noise_sigma;
    
    % [UPDATED] 2. 陀螺仪零偏 (读取 sens_gyro_bias)
    % 仅加在角速度项 (p, q, r) -> 索引 10:12
    bias_vec = zeros(12, 1);
    bias_vec(10:12) = params.sens_gyro_bias;
    
    % 3. 合成测量值
    state_meas = state_true + bias_vec + noise;
end