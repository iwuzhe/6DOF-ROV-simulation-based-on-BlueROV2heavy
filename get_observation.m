function obs = get_observation(state_meas, state_ref, dist_est)
    % 1. 计算误差
    p_err = state_ref(1:3) - state_meas(1:3);
    % % R_err = R_des^T * R_act - R_act^T * R_des
    att = eul2rotm_custom(state_meas(4), state_meas(5), state_meas(6));
    att_ref = eul2rotm_custom(state_ref(4), state_ref(5), state_ref(6));
    att_err = att_ref' * att - att' * att_ref;
    e_att_body = 0.5 * vee_map(att_err);
    v_err = state_ref(7:12) - state_meas(7:12);

    % 2. 归一化 (非常重要! RL 对数值范围敏感)
    % 假设位置误差最大 1m, 速度误差 1m/s, 扰动 50N
    obs = [
        p_err;           % [3]
        e_att_body;         % [3]
        v_err;           % [6]
        state_meas(4:6); % [3] 当前姿态 (感知重力方向)
        dist_est / 50.0  % [6] 归一化的扰动估计
    ]; 
    % 总维度: 3+3+6+3+6 = 21维
end

function R = eul2rotm_custom(phi, theta, psi)
    cph = cos(phi); sph = sin(phi);
    cth = cos(theta); sth = sin(theta);
    cps = cos(psi); sps = sin(psi);
    R = [cps*cth, cps*sth*sph-sps*cph, cps*sth*cph+sps*sph;
         sps*cth, sps*sth*sph+cps*cph, sps*sth*cph-cps*sph;
         -sth,    cth*sph,             cth*cph];
end

function v = vee_map(S)
    v = [-S(2,3); S(1,3); -S(1,2)];
end