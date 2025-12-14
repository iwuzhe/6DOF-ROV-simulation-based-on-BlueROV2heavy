import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from ResidualROVEnv import ResidualROVEnv

# === PPO Hyperparams ===
NUM_ENVS = 128
NUM_STEPS = 2048
TOTAL_TIMESTEPS = 5e7
MINIBATCH_SIZE = 4096
PPO_EPOCHS = 4
CLIP_COEF = 0.2
ENT_COEF = 0.01
LR = 1e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95

OBS_DIM = 33
ACT_DIM = 6


class RunningMeanStd(nn.Module):
    def __init__(self, shape, epsilon=1e-4, device='cpu'):
        super().__init__()
        self.register_buffer('mean', torch.zeros(shape, device=device))
        self.register_buffer('var', torch.ones(shape, device=device))
        self.register_buffer('count', torch.tensor(epsilon, device=device))

    def update(self, x):
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0, unbiased=False)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x):
        return (x - self.mean) / torch.sqrt(self.var + 1e-8)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh()
        )
        self.actor_mean = nn.Linear(128, act_dim)
        self.actor_activation = nn.Tanh()
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.log_std = nn.Parameter(torch.ones(1, act_dim) * -0.5)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        hidden = self.feature_extractor(x)
        mean = self.actor_activation(self.actor_mean(hidden))
        std = self.log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        if action is None:
            action = dist.sample()
        action_log_prob = dist.log_prob(action).sum(1)
        entropy = dist.entropy().sum(1)
        value = self.critic(x)
        return action, action_log_prob, entropy, value


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}...")

    env = ResidualROVEnv(num_envs=NUM_ENVS, device=device)
    agent = ActorCritic(OBS_DIM, ACT_DIM).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LR)
    obs_normalizer = RunningMeanStd((OBS_DIM,), device=device)


    PRETRAINED_PATH = "ppo_rov_agent_stage2.pth"
    FREEZE_NORMALIZER = False  # 新增标志
    if os.path.exists(PRETRAINED_PATH):
        print(f">>> 加载已有模型: {PRETRAINED_PATH}...")
        checkpoint = torch.load(PRETRAINED_PATH, map_location=device)
            
        agent.load_state_dict(checkpoint['model_state_dict'])
        obs_normalizer.mean = checkpoint['obs_norm_mean']
        obs_normalizer.var = checkpoint['obs_norm_var']
        obs_normalizer.count = checkpoint.get('obs_norm_count', torch.tensor(1e7, device=device))
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(">>> Optimizer已加载")
        FREEZE_NORMALIZER = True  # ← 冻结！
        print(">>> 加载成功! (归一化已冻结)")
    else:
        print(">>> 未找到模型")

    obs = env.reset()
    num_updates = int(TOTAL_TIMESTEPS // (NUM_ENVS * NUM_STEPS))

    for update in range(num_updates):
        b_obs, b_acts, b_logprobs, b_rews, b_dones, b_vals = [], [], [], [], [], []

        # === 1. 数据采集 ===
        for step in range(NUM_STEPS):
            if not FREEZE_NORMALIZER:  # ← 只在从头训练时更新
                obs_normalizer.update(obs)
            norm_obs = obs_normalizer.normalize(obs)

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(norm_obs)

            next_obs, reward, done, _ = env.step(action)

            b_obs.append(norm_obs)
            b_acts.append(action)
            b_logprobs.append(logprob)
            b_rews.append(reward)
            b_dones.append(done)
            b_vals.append(value.flatten())
            obs = next_obs

        # === [关键修复] ===
        # 1. 先保存 Batch 结束时的真实状态 (用于计算正确的 Bootstrap Value)
        last_obs = obs.clone()

        # 2. 然后再重置环境 (用于下一个 Batch 的 Domain Randomization)
        # 这样不会打断本次 GAE 计算的连贯性
        obs = env.reset()
        
        # === 2. 数据处理 ===
        b_obs = torch.stack(b_obs)
        b_acts = torch.stack(b_acts)
        b_logprobs = torch.stack(b_logprobs)
        b_rews = torch.stack(b_rews)
        b_vals = torch.stack(b_vals)

        with torch.no_grad():
            # 3. 使用保存的 last_obs 计算 next_value
            obs_normalizer.update(last_obs)
            norm_last = obs_normalizer.normalize(last_obs)
            next_value = agent.get_value(norm_last).reshape(1, -1)

            advantages = torch.zeros_like(b_rews)
            lastgaelam = 0
            for t in reversed(range(NUM_STEPS)):
                if t == NUM_STEPS - 1:
                    # 如果是最后一步，nextvalues 取我们刚才算出的正确引导值
                    nextnonterminal = 1.0
                    nextvalues = next_value
                else:
                    # 如果中间死了 (done=True)，nextnonterminal=0，阻断回传
                    nextnonterminal = 1.0 - b_dones[t].float()
                    nextvalues = b_vals[t + 1]

                delta = b_rews[t] + GAMMA * nextvalues * nextnonterminal - b_vals[t]
                advantages[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam

            returns = advantages + b_vals

        b_obs = b_obs.reshape(-1, OBS_DIM)
        b_acts = b_acts.reshape(-1, ACT_DIM)
        b_logprobs = b_logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = b_vals.reshape(-1)

        # Advantage Normalization
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # === 3. PPO 更新 ===
        b_inds = np.arange(NUM_ENVS * NUM_STEPS)
        for epoch in range(PPO_EPOCHS):
            np.random.shuffle(b_inds)
            for start in range(0, NUM_ENVS * NUM_STEPS, MINIBATCH_SIZE):
                end = start + MINIBATCH_SIZE
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_acts[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_adv = b_advantages[mb_inds]

                pg_loss = -torch.min(mb_adv * ratio, mb_adv * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)).mean()
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - ENT_COEF * entropy_loss + 0.5 * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

        # === 4. 日志与保存 ===
        if (update + 1) % 1 == 0:
            print(f"Update {update + 1}/{num_updates} | Reward: {b_rews.mean().item():.4f} | Loss: {loss.item():.4f}")

        if (update + 1) % 5 == 0:
            torch.save({
                'model_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'obs_norm_mean': obs_normalizer.mean,
                'obs_norm_var': obs_normalizer.var,
                'obs_norm_count': obs_normalizer.count,
            }, f"ppo_rov_agent_{update + 1}.pth")

            # 保存 latest 用于测试
            torch.save({
                'model_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'obs_norm_mean': obs_normalizer.mean,
                'obs_norm_var': obs_normalizer.var
            }, "ppo_rov_agent_final.pth")
            print(f">>> Model saved: ppo_rov_agent_{update + 1}.pth")


if __name__ == "__main__":
    train()