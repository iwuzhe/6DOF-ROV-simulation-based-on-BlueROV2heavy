import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from ResidualROVEnv import ResidualROVEnv

# === PPO Hyperparameters ===
NUM_ENVS = 4096  # 并行环境数 (越大训练越快越稳)
NUM_STEPS = 200  # 每个环境采样的步数 (trajectory length)
TOTAL_TIMESTEPS = 5e7  # 总训练步数
MINIBATCH_SIZE = 512  # PPO 更新的 batch size
PPO_EPOCHS = 5  # 每次更新循环次数
CLIP_COEF = 0.2  # PPO clip range
ENT_COEF = 0.001  # 熵正则化系数 (鼓励探索)
LR = 3e-4  # 学习率
GAMMA = 0.99  # 折扣因子
GAE_LAMBDA = 0.95  # GAE 参数


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(),
            nn.Linear(128, act_dim), nn.Tanh()  # 输出范围 [-1, 1]
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.log_std = nn.Parameter(torch.zeros(1, act_dim))  # 可学习的标准差

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        mean = self.actor(x)
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

    # 1. Init Env
    env = ResidualROVEnv(num_envs=NUM_ENVS, device=device)
    obs_dim = 30  # [e_p(3), e_att(3), e_v(3), e_w(3), est_dist(6), est_state(12)]
    act_dim = 6  # [Fx, Fy, Fz, Tx, Ty, Tz]

    # 2. Init Network
    agent = ActorCritic(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LR)

    # 3. Training Loop
    obs = env.reset()
    global_step = 0
    num_updates = int(TOTAL_TIMESTEPS // (NUM_ENVS * NUM_STEPS))

    for update in range(num_updates):
        # --- Data Collection ---
        b_obs, b_acts, b_logprobs, b_rews, b_dones, b_vals = [], [], [], [], [], []

        for step in range(NUM_STEPS):
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(obs)

            next_obs, reward, done, _ = env.step(action)

            # Store data
            b_obs.append(obs)
            b_acts.append(action)
            b_logprobs.append(logprob)
            b_rews.append(reward)
            b_dones.append(done)
            b_vals.append(value.flatten())

            obs = next_obs

            # 这里的环境是无限长的，每 NUM_STEPS 强制 reset 一次以防止误差发散太远
            # 或者在 env 内部检测时间重置。这里简化处理，不做中途 reset。

        # 强制重置环境 (模拟 Episode 结束)
        obs = env.reset()

        # --- GAE Calculation ---
        b_obs = torch.stack(b_obs)
        b_acts = torch.stack(b_acts)
        b_logprobs = torch.stack(b_logprobs)
        b_rews = torch.stack(b_rews)
        b_vals = torch.stack(b_vals)

        with torch.no_grad():
            next_value = agent.get_value(obs).reshape(1, -1)
            advantages = torch.zeros_like(b_rews)
            lastgaelam = 0
            for t in reversed(range(NUM_STEPS)):
                if t == NUM_STEPS - 1:
                    nextnonterminal = 0.0  # Episode 结束
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0
                    nextvalues = b_vals[t + 1]

                delta = b_rews[t] + GAMMA * nextvalues * nextnonterminal - b_vals[t]
                advantages[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam

            returns = advantages + b_vals

        # Flatten batch
        b_obs = b_obs.reshape(-1, obs_dim)
        b_acts = b_acts.reshape(-1, act_dim)
        b_logprobs = b_logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = b_vals.reshape(-1)

        # --- PPO Update ---
        b_inds = np.arange(NUM_ENVS * NUM_STEPS)
        for epoch in range(PPO_EPOCHS):
            np.random.shuffle(b_inds)
            for start in range(0, NUM_ENVS * NUM_STEPS, MINIBATCH_SIZE):
                end = start + MINIBATCH_SIZE
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_acts[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                # Normalize advantages
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy Loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value Loss
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # Entropy Loss
                entropy_loss = entropy.mean()

                loss = pg_loss - ENT_COEF * entropy_loss + 0.5 * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

        print(f"Update {update + 1}/{num_updates} | Reward Mean: {b_rews.mean().item():.4f} | Loss: {loss.item():.4f}")

        # Save Model periodically
        if (update + 1) % 1 == 0:
            torch.save(agent.state_dict(), f"ppo_rov_agent_{update + 1}.pth")
            print("Model saved.")
    final_path = "ppo_rov_agent_final.pth"
    torch.save(agent.state_dict(), final_path)
    print(f"Training Done! Final model saved to: {final_path}")


if __name__ == "__main__":
    train()