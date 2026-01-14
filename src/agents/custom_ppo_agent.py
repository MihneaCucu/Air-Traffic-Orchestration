import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
import datetime

class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        shared_features = self.shared(x)
        action_probs = self.actor(shared_features)
        state_value = self.critic(shared_features)
        return action_probs, state_value
    
    def get_action_probs(self, x):
        shared_features = self.shared(x)
        return self.actor(shared_features)
    
    def get_value(self, x):
        shared_features = self.shared(x)
        return self.critic(shared_features)

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def push(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def get(self):
        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.rewards),
            np.array(self.values),
            np.array(self.log_probs),
            np.array(self.dones)
        )
    
    def __len__(self):
        return len(self.states)

class CustomPPO:
    def __init__(self, env, learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_range=0.2, n_steps=2048, batch_size=64, n_epochs=10,
                 ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
                 log_dir="atc_logs", device=None, seed=None):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.seed = seed
        
        # Set seeds for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        
        self.policy = ActorCritic(self.obs_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        
        self.rollout_buffer = RolloutBuffer()
        
        # Don't create writer in __init__, will be created in learn()
        self.writer = None
        self.log_dir = log_dir
        self.steps_done = 0
    
    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_probs, value = self.policy(state_t)
            
            if deterministic:
                action = action_probs.argmax().item()
                log_prob = torch.log(action_probs[0, action])
            else:
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                action = action.item()
            
            return action, value.item(), log_prob.item()
    
    def compute_gae(self, rewards, values, dones, next_value):
        """Compute Generalized Advantage Estimation"""
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value_t * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
        
        returns = advantages + values
        return advantages, returns
    
    def update(self, states, actions, old_log_probs, returns, advantages):
        """Update policy using PPO loss"""
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        old_log_probs_t = torch.FloatTensor(old_log_probs).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        
        # Multiple epochs of optimization
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        n_updates = 0
        
        for epoch in range(self.n_epochs):
            # Create random minibatches
            indices = np.random.permutation(len(states))
            
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]
                
                # Get current policy predictions
                action_probs, values = self.policy(states_t[batch_idx])
                dist = torch.distributions.Categorical(action_probs)
                
                # Calculate losses
                log_probs = dist.log_prob(actions_t[batch_idx])
                entropy = dist.entropy().mean()
                
                # Policy loss (PPO clipped objective)
                ratio = torch.exp(log_probs - old_log_probs_t[batch_idx])
                surr1 = ratio * advantages_t[batch_idx]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages_t[batch_idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.MSELoss()(values.squeeze(), returns_t[batch_idx])
                
                # Total loss
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy.item()
                n_updates += 1
        
        return (total_policy_loss / n_updates, 
                total_value_loss / n_updates, 
                total_entropy_loss / n_updates)
    
    def learn(self, total_timesteps, log_name=None):
        # Create writer with custom name if provided
        if log_name:
            log_path = os.path.join(self.log_dir, log_name)
        else:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            log_path = os.path.join(self.log_dir, f"CustomPPO_{current_time}")
        
        self.writer = SummaryWriter(log_dir=log_path)
        
        state, _ = self.env.reset()
        episode_reward = 0
        episode_steps = 0
        episodes = 0
        
        while self.steps_done < total_timesteps:
            # Collect rollout
            self.rollout_buffer.clear()
            
            for _ in range(self.n_steps):
                self.steps_done += 1
                
                action, value, log_prob = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                self.rollout_buffer.push(state, action, reward, value, log_prob, done)
                
                state = next_state
                episode_reward += reward
                episode_steps += 1
                
                if done:
                    episodes += 1
                    self.writer.add_scalar("rollout/ep_rew_mean", episode_reward, self.steps_done)
                    
                    print(f"Step {self.steps_done}/{total_timesteps} | Episode {episodes} | Reward: {episode_reward:.2f}")
                    
                    state, _ = self.env.reset()
                    episode_reward = 0
                    episode_steps = 0
                
                if self.steps_done >= total_timesteps:
                    break
            
            # Get rollout data
            states, actions, rewards, values, log_probs, dones = self.rollout_buffer.get()
            
            # Compute next value for GAE
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                _, next_value = self.policy(state_t)
                next_value = next_value.item()
            
            # Compute advantages and returns
            advantages, returns = self.compute_gae(rewards, values, dones, next_value)
            
            # Update policy
            policy_loss, value_loss, entropy = self.update(states, actions, log_probs, returns, advantages)
            
            # Log training metrics
            self.writer.add_scalar("train/policy_loss", policy_loss, self.steps_done)
            self.writer.add_scalar("train/value_loss", value_loss, self.steps_done)
            self.writer.add_scalar("train/entropy", entropy, self.steps_done)
        
        # Close writer after training
        if self.writer:
            self.writer.close()
    
    def save(self, path):
        torch.save(self.policy.state_dict(), path)
    
    def load(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
        self.policy.eval()
    
    def predict(self, obs, deterministic=True):
        action, _, _ = self.select_action(obs, deterministic=deterministic)
        return action, None
