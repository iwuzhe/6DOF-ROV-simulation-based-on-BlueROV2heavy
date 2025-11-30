%% === 1. 环境初始化 ===
clear; clc;

% 加载基础参数
init_blueROV2Heavy; 

%% 2. 环境定义
mdl = 'BlueROV2_Heavy_6DOF'; 
% 检查模型是否加载，未加载则打开
if ~bdIsLoaded(mdl)
    open_system(mdl);
end

set_param(mdl, 'SimulationMode', 'accelerator');
set_param(mdl, 'FastRestart', 'on'); % 确保开启快速重启 (可选，默认通常是开启的)
agent_blk = [mdl '/RLsystem/RLAgent'];

% 观测空间 (根据 get_observation 的输出维度)
obsInfo = rlNumericSpec([21 1]); 
obsInfo.Name = 'Observation';

% 动作空间 (RL 输出的 6维残差力矩)
actInfo = rlNumericSpec([6 1]);
actInfo.LowerLimit = -40; % 限制 RL 的修正幅度
actInfo.UpperLimit =  40;
actInfo.Name = 'Residual_Action';

% 创建 Simulink 环境接口
env = rlSimulinkEnv(mdl, agent_blk, obsInfo, actInfo);

% 定义 Reset 函数 (每次重开时随机化参数)
env.ResetFcn = @(in) my_reset_function(in); 

%% 3. 智能体 (Agent) 定义 - 推荐使用 TD3 或 SAC (处理连续动作)
% 默认网络结构
agent = rlTD3Agent(obsInfo, actInfo);

% 
% agentOptions = rlTD3AgentOptions;
% agentOptions.SampleTime = 0.01; % 与仿真步长一致
% agentOptions.DiscountFactor = 0.99;
% agent = rlTD3Agent(obsInfo, actInfo, agentOptions);

%% 4. 训练配置
trainOpts = rlTrainingOptions(...
    'MaxEpisodes', 500, ...
    'MaxStepsPerEpisode', 1000, ... % 10秒
    'ScoreAveragingWindow', 20, ...
    'Plots', 'training-progress', ...
    'StopTrainingCriteria', 'AverageReward', ...
    'StopTrainingValue', 500); 

%% 4. 开始训练
trainingStats = train(agent, env, trainOpts);