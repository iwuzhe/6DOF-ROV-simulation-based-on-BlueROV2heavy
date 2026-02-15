function plot_results(out)
% out.state_actual_ts : timeseries, 数据为 12×1
% out.state_ref       : timeseries, 同维度

state_act = ts_to_matrix(out.state_actual_ts);
state_ref = ts_to_matrix(out.state_ref);

t = out.state_actual_ts.Time;  % 统一以实际状态的时间轴为准
state_ref_interp = interp1(out.state_ref.Time, state_ref, t, 'linear', 'extrap');

pos_act = state_act(:,1:3);    att_act = wrapToPi(state_act(:,4:6));
pos_ref = state_ref_interp(:,1:3);
att_ref = wrapToPi(state_ref_interp(:,4:6));

figure('Name','3D Trajectory','Color','w');
plot3(pos_ref(:,1),pos_ref(:,2),pos_ref(:,3),'g--','LineWidth',1.2); hold on;
plot3(pos_act(:,1),pos_act(:,2),pos_act(:,3),'b-','LineWidth',1.4);
% 标记起点
plot3(pos_ref(1,1),pos_ref(1,2),pos_ref(1,3),'go','MarkerSize',10,'MarkerFaceColor','g','LineWidth',2);
plot3(pos_act(1,1),pos_act(1,2),pos_act(1,3),'bs','MarkerSize',10,'MarkerFaceColor','b','LineWidth',2);
% 标记终点
plot3(pos_ref(end,1),pos_ref(end,2),pos_ref(end,3),'g^','MarkerSize',10,'MarkerFaceColor','g','LineWidth',2);
plot3(pos_act(end,1),pos_act(end,2),pos_act(end,3),'b^','MarkerSize',10,'MarkerFaceColor','b','LineWidth',2);
grid on; axis equal;
xlabel('X [m]'); ylabel('Y [m]'); zlabel('Z [m]');
legend('Reference','Actual','Ref Start','Act Start','Ref End','Act End','Location','best'); title('ROV Trajectory');

labels = {'X','Y','Z','Roll','Pitch','Yaw'};
figure('Name','State Tracking','Color','w');
for i = 1:3
    subplot(3,2,2*i-1);
    plot(t,pos_ref(:,i),'g--',t,pos_act(:,i),'b-','LineWidth',1.2);
    grid on; ylabel([labels{i} ' [m]']);
    if i==3, xlabel('Time [s]'); end
end
for i = 1:3
    subplot(3,2,2*i);
    plot(t,att_ref(:,i),'g--',t,att_act(:,i),'b-','LineWidth',1.2);
    grid on; ylabel([labels{i+3} ' [rad]']);
    ylim([-pi pi]); yticks([-pi -pi/2 0 pi/2 pi]);
    if i==3, xlabel('Time [s]'); end
end
end

function mat = ts_to_matrix(ts)
mat = squeeze(ts.Data);
ntime = numel(ts.Time);
if size(mat,1) ~= ntime && size(mat,2) == ntime
    mat = mat.';  % 转置为 (time x state)
elseif size(mat,1) ~= ntime && size(mat,2) ~= ntime
    error('ts_to_matrix:SizeMismatch','timeseries 数据维度与时间长度不一致。');
end
end