function [r, is_done] = calc_reward(e_pos, e_att, tau_total, tau_res)
    % 权重定义
    w_pos = 2.0;
    w_att = 1.0;
    w_energy = 0.001; % 惩罚总能耗
    w_action = 0;  % 惩罚 RL 乱动 (Regularization)

    % 1. 跟踪奖励 (误差越小奖励越高，最大为1)
    r_pos = exp(-w_pos * norm(e_pos)^2);
    r_att = exp(-w_att * norm(e_att)^2);

    % 2. 惩罚项
    p_energy = w_energy * norm(tau_total)^2;
    p_smooth = w_action * norm(tau_res)^2; % 希望 RL 输出尽可能小

    % 总奖励
    r = r_pos + r_att - p_energy - p_smooth;

    % 3. 终止条件 (IsDone)
    % 如果误差太大，强制结束，给一个巨大惩罚
    if norm(e_pos) > 2.0 || abs(e_att(2)) > deg2rad(60)
        is_done = 1;
        r = r - 100; % 死亡惩罚
    else
        is_done = 0;
    end
end