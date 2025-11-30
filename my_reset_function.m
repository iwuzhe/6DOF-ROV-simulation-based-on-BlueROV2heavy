function in = my_reset_function(in)
    % 1. 随机化物理参数
    m_rand = 14.5 * (0.9 + 0.2*rand); % 质量 +/- 10%
    in = setVariable(in, 'params.m', m_rand);
    
    % 2. 随机化水动力阻力 (这是 RL 最需要克服的)
    Xu_rand = -4.03 * (0.5 + rand); % 阻力系数 50% ~ 150%
    in = setVariable(in, 'params.Xu', Xu_rand);
    
    % 3. 随机化洋流 (最重要的扰动)
    current_speed = 0.5 * rand; % 0 ~ 0.5 m/s
    % 必须把这个写入 params 供 rov_plant 使用
    % 注意：你需要修改 init 脚本，允许 params 部分字段在 workspace 被覆盖
    assignin('base', 'current_speed_sim', current_speed); 
    
    % 4. 随机化初始位置 (防止过拟合某一条轨迹)
    % init_pos = [0.1*randn; 0.1*randn; 2];
    % assignin('base', 'initial_state', [init_pos; zeros(9,1)]);
end