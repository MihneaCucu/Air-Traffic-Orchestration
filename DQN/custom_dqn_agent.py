import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os
from torch.utils.tensorboard import SummaryWriter
import datetime

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), action, reward, np.array(next_state), done

    def __len__(self):
        return len(self.buffer)

class CustomDQN:
    def __init__(self, env, learning_rate=1e-3, gamma=0.99, batch_size=64, buffer_size=10000, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, 
                 target_update_freq=1000, log_dir="atc_logs", device=None, seed=None):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        if device:
             self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.q_net = QNetwork(self.obs_dim, self.action_dim).to(self.device)
        self.target_net = QNetwork(self.obs_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.learning_rate)
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        self.writer = None
        self.log_dir = log_dir
        self.steps_done = 0

    def select_action(self, state, deterministic=False):
        if not deterministic and random.random() < self.epsilon:
            return self.env.action_space.sample()
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_net(state_t)
            return q_values.argmax().item()

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        state_t = torch.FloatTensor(state).to(self.device)
        action_t = torch.LongTensor(action).unsqueeze(1).to(self.device)
        reward_t = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state_t = torch.FloatTensor(next_state).to(self.device)
        done_t = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        q_values = self.q_net(state_t).gather(1, action_t)

        with torch.no_grad():
            next_q_values = self.target_net(next_state_t).max(1)[0].unsqueeze(1)
            target_q_values = reward_t + (1 - done_t) * self.gamma * next_q_values

        loss = nn.MSELoss()(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def learn(self, total_timesteps, log_name=None):
        if log_name:
            log_path = os.path.join(self.log_dir, log_name)
        else:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            log_path = os.path.join(self.log_dir, f"CustomDQN_{current_time}")
        self.writer = SummaryWriter(log_dir=log_path)
        state, _ = self.env.reset()
        episode_reward = 0
        episode_steps = 0
        episodes = 0
        for t in range(1, total_timesteps + 1):
            self.steps_done += 1
            action = self.select_action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            self.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            episode_steps += 1

            loss = self.update()

            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

            if t % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())

            if done:
                episodes += 1
                self.writer.add_scalar("rollout/ep_rew_mean", episode_reward, self.steps_done)
                self.writer.add_scalar("train/epsilon", self.epsilon, self.steps_done)
                if loss:
                    self.writer.add_scalar("train/loss", loss, self.steps_done)
                print(f"Step {t}/{total_timesteps} | Episode {episodes} | Reward: {episode_reward:.2f} | Epsilon: {self.epsilon:.4f}")
                state, _ = self.env.reset()
                episode_reward = 0
                episode_steps = 0
        if self.writer:
            self.writer.close()

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.q_net.eval()
        self.target_net.load_state_dict(self.q_net.state_dict())

    def predict(self, obs, deterministic=True):
        action = self.select_action(obs, deterministic=deterministic)
        return action, None
