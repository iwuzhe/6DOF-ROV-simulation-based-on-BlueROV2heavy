function state_meas = sensor_model(state_true, params)
    % state_true: [eta; nu] 真实状态
    
    % 1. 基础高斯白噪声
    noise = randn(size(state_true)) .* params.sensor_noise_sigma;
    
    % 2. 陀螺仪零偏 (Bias) - 模拟积分漂移
    % 可以在 init 文件中随机生成一个固定的 bias
    bias_gyro = params.sensor_gyro_bias; 
    
    state_meas = state_true + noise;
    state_meas(10:12) = state_meas(10:12) + bias_gyro; % 加在角速度上
end