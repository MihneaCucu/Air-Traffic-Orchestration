import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
from torch.utils.tensorboard import SummaryWriter
import datetime

class ActorCriticNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=128):
        super(ActorCriticNetwork, self).__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        shared = self.shared(x)
        logits = self.actor(shared)
        value = self.critic(shared)
        return logits, value


class A2CAgent:
    def __init__(self, env, learning_rate=1e-3, gamma=0.99, entropy_coef=0.1, 
                 value_coef=0.5, max_grad_norm=0.5, log_dir="atc_logs", 
                 device=None, seed=None):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
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
        
        self.net = ActorCriticNetwork(self.obs_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5
        )
        
        self.writer = None
        self.log_dir = log_dir
        self.steps_done = 0

    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            logits, value = self.net(state_t)
            
            if deterministic:
                action = logits.argmax(dim=1).item()
            else:
                probs = torch.softmax(logits, dim=1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample().item()
            
            return action

    def get_action_distribution(self, state):
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            logits, _ = self.net(state_t)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        return probs

    def compute_returns(self, rewards, dones, next_value, gamma):
        returns = []
        R = next_value
        
        for t in reversed(range(len(rewards))):
            R = rewards[t] + gamma * R * (1 - dones[t])
            returns.insert(0, R)
        
        return np.array(returns)

    def update(self, states, actions, rewards, next_states, dones):
        T = len(rewards)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        logits, values = self.net(states)
        values = values.squeeze(1) 

        with torch.no_grad():
            last_next = torch.FloatTensor(next_states[-1]).unsqueeze(0).to(self.device)
            _, last_value = self.net(last_next)
            last_value = last_value.squeeze(1).squeeze(0).item()

        returns = self.compute_returns(rewards, dones, last_value, self.gamma)
        returns = torch.FloatTensor(returns).to(self.device).squeeze()

        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        log_probs = torch.log_softmax(logits, dim=1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        actor_loss = -(action_log_probs * advantages.detach()).mean()

        probs = torch.softmax(logits, dim=1)
        entropy = -(probs * log_probs).sum(dim=1).mean()

        critic_loss = 0.5 * (advantages.pow(2)).mean()

        total_loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return total_loss.item(), actor_loss.item(), critic_loss.item(), entropy.item()

    def learn(self, total_timesteps, n_steps=20, log_name=None):
        if log_name:
            log_path = os.path.join(self.log_dir, log_name)
        else:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            log_path = os.path.join(self.log_dir, f"A2C_{current_time}")
        
        self.writer = SummaryWriter(log_dir=log_path)
        
        state, _ = self.env.reset(seed=self.seed)
        episode_reward = 0
        episodes = 0
        
        for t in range(1, total_timesteps + 1):
            self.steps_done += 1
            
            states, actions, rewards, next_states, dones = [], [], [], [], []
            action_counts = {i: 0 for i in range(self.action_dim)}
            
            for _ in range(n_steps):
                action = self.select_action(state, deterministic=False)
                action_counts[action] += 1
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)
                
                episode_reward += reward
                
                if done:
                    episodes += 1
                    self.writer.add_scalar("rollout/ep_rew_mean", episode_reward, self.steps_done)
                    print(f"Step {self.steps_done}/{total_timesteps} | Episode {episodes} | Reward: {episode_reward:.2f}")
                    
                    state, _ = self.env.reset()
                    episode_reward = 0
                else:
                    state = next_state
            
            total_loss, actor_loss, critic_loss, entropy = self.update(
                states, actions, rewards, next_states, dones
            )
            
            self.writer.add_scalar("train/total_loss", total_loss, self.steps_done)
            self.writer.add_scalar("train/actor_loss", actor_loss, self.steps_done)
            self.writer.add_scalar("train/critic_loss", critic_loss, self.steps_done)
            self.writer.add_scalar("train/entropy", entropy, self.steps_done)
            for a_idx, cnt in action_counts.items():
                self.writer.add_scalar(f"policy/action_{a_idx}_count", cnt, self.steps_done)
        
        if self.writer:
            self.writer.close()

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path, map_location=self.device))
        self.net.eval()

    def predict(self, obs, deterministic=True):
        action = self.select_action(obs, deterministic=deterministic)
        return action, None
