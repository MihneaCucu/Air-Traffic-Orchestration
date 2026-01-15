import os
import math
import random
import datetime
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter



class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, sigma0: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.sigma0 = sigma0
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        self.weight_sigma.data.fill_(self.sigma0 / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma0 / math.sqrt(self.out_features))

    @staticmethod
    def _scale_noise(size: int, device: torch.device) -> torch.Tensor:
        x = torch.randn(size, device=device)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self) -> None:
        eps_in = self._scale_noise(self.in_features, self.weight_epsilon.device)
        eps_out = self._scale_noise(self.out_features, self.weight_epsilon.device)
        self.weight_epsilon.copy_(eps_out.ger(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return torch.nn.functional.linear(x, weight, bias)



class RainbowDuelingC51Network(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, atoms: int = 51,
                 v_min: float = -50.0, v_max: float = 150.0, hidden_dim: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.atoms = atoms
        self.v_min = v_min
        self.v_max = v_max

        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.value = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            NoisyLinear(hidden_dim, atoms),
        )
        self.advantage = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            NoisyLinear(hidden_dim, action_dim * atoms),
        )

        self.register_buffer("support", torch.linspace(v_min, v_max, atoms))
        self.delta_z = (v_max - v_min) / (atoms - 1)

    def reset_noise(self) -> None:
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.feature(x)

        v = self.value(h)
        a = self.advantage(h)
        a = a.view(-1, self.action_dim, self.atoms)

        v = v.view(-1, 1, self.atoms)
        q_atoms = v + (a - a.mean(dim=1, keepdim=True))

        probs = torch.softmax(q_atoms, dim=-1)
        return probs

    def q_values(self, x: torch.Tensor) -> torch.Tensor:
        probs = self.forward(x)
        q = torch.sum(probs * self.support, dim=-1)
        return q



class SumTree:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.tree = np.zeros(2 * self.capacity, dtype=np.float32)
        self.data = np.empty(self.capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def total(self) -> float:
        return float(self.tree[1])

    def add(self, p: float, data: object) -> None:
        idx = self.write + self.capacity
        self.data[self.write] = data
        self.update(idx, p)

        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx: int, p: float) -> None:
        change = p - self.tree[idx]
        self.tree[idx] = p
        idx //= 2
        while idx >= 1:
            self.tree[idx] += change
            idx //= 2

    def get(self, s: float) -> Tuple[int, float, object]:
        idx = 1
        while idx < self.capacity:
            left = 2 * idx
            right = left + 1
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        data_idx = idx - self.capacity
        return idx, float(self.tree[idx]), self.data[data_idx]


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = int(capacity)
        self.alpha = alpha
        self.eps = 1e-6
        self.tree = SumTree(self.capacity)
        self.max_priority = 1.0

    def __len__(self) -> int:
        return self.tree.n_entries

    def push(self, transition: object) -> None:
        p = self.max_priority
        self.tree.add(p, transition)

    def sample(self, batch_size: int, beta: float):
        total = self.tree.total()
        if total <= 0.0:
            raise RuntimeError("PER: cannot sample because total priority is 0.")

        batch = []
        idxs = []
        priorities = []

        segment = total / batch_size

        for i in range(batch_size):
            a = segment * i

            s = a + random.random() * segment  

            idx, p, data = self.tree.get(s)

            tries = 0
            while (data is None or p <= 0.0) and tries < 10:
                s = random.random() * total
                idx, p, data = self.tree.get(s)
                tries += 1

            if data is None:
                data = self.tree.data[random.randrange(self.tree.n_entries)]

            batch.append(data)
            idxs.append(idx)
            priorities.append(p)

        priorities = np.array(priorities, dtype=np.float32)

        probs = priorities / max(total, 1e-8)
        probs = np.clip(probs, 1e-8, 1.0)

        is_weights = (len(self) * probs) ** (-beta)
        is_weights /= (is_weights.max() + 1e-8)

        return idxs, np.array(batch, dtype=object), is_weights.astype(np.float32)

    def update_priorities(self, idxs: List[int], priorities: np.ndarray) -> None:
        priorities = np.asarray(priorities, dtype=np.float32)
        priorities = np.maximum(priorities, self.eps)
        self.max_priority = max(self.max_priority, float(priorities.max()))

        for idx, p in zip(idxs, priorities):
            self.tree.update(idx, float(p) ** self.alpha)



@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class NStepAccumulator:
    def __init__(self, n: int, gamma: float):
        self.n = int(n)
        self.gamma = float(gamma)
        self.buffer: Deque[Transition] = deque(maxlen=self.n)

    def reset(self) -> None:
        self.buffer.clear()

    def push(self, t: Transition) -> Optional[Transition]:
        self.buffer.append(t)
        if len(self.buffer) < self.n:
            return None

        R = 0.0
        for i, tr in enumerate(self.buffer):
            R += (self.gamma ** i) * tr.reward
            if tr.done:
                break

        first = self.buffer[0]
        last = self.buffer[-1]
        done = any(tr.done for tr in self.buffer)
        if done:
            for tr in self.buffer:
                if tr.done:
                    last = tr
                    break

        return Transition(
            state=first.state,
            action=first.action,
            reward=R,
            next_state=last.next_state,
            done=done
        )

    def flush(self) -> List[Transition]:
        out = []
        while len(self.buffer) > 0:
            R = 0.0
            for i, tr in enumerate(self.buffer):
                R += (self.gamma ** i) * tr.reward
                if tr.done:
                    break
            first = self.buffer[0]
            last = self.buffer[-1]
            done = any(tr.done for tr in self.buffer)
            if done:
                for tr in self.buffer:
                    if tr.done:
                        last = tr
                        break
            out.append(Transition(first.state, first.action, R, last.next_state, done))
            self.buffer.popleft()
        return out



class RainbowDQN:

    def __init__(
        self,
        env,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        batch_size: int = 64,
        buffer_size: int = 100000,
        target_update_freq: int = 2000,
        train_start: int = 2000,
        train_freq: int = 1,
        gradient_steps: int = 1,
        n_step: int = 3,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_frames: int = 500000,
        atoms: int = 51,
        v_min: float = -50.0,
        v_max: float = 150.0,
        hidden_dim: int = 128,
        log_dir: str = "atc_logs",
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.target_update_freq = target_update_freq
        self.train_start = train_start
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps

        self.n_step = n_step
        self.n_gamma = gamma ** n_step

        self.per_alpha = per_alpha
        self.per_beta_start = per_beta_start
        self.per_beta_frames = max(1, int(per_beta_frames))

        self.atoms = atoms
        self.v_min = v_min
        self.v_max = v_max

        self.log_dir = log_dir
        self.writer: Optional[SummaryWriter] = None
        self.steps_done = 0
        self.episodes = 0

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.obs_dim = int(env.observation_space.shape[0])
        self.action_dim = int(env.action_space.n)

        self.q_net = RainbowDuelingC51Network(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            atoms=self.atoms,
            v_min=self.v_min,
            v_max=self.v_max,
            hidden_dim=hidden_dim,
        ).to(self.device)

        self.target_net = RainbowDuelingC51Network(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            atoms=self.atoms,
            v_min=self.v_min,
            v_max=self.v_max,
            hidden_dim=hidden_dim,
        ).to(self.device)

        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.learning_rate)

        self.replay = PrioritizedReplayBuffer(capacity=self.buffer_size, alpha=self.per_alpha)
        self.nstep_acc = NStepAccumulator(n=self.n_step, gamma=self.gamma)

        self.support = self.q_net.support.to(self.device)
        self.delta_z = self.q_net.delta_z

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        if deterministic:
            self.q_net.eval()
            with torch.no_grad():
                q = self.q_net.q_values(state_t)
                action = int(torch.argmax(q, dim=1).item())
            self.q_net.train()
            return action

        self.q_net.train()
        self.q_net.reset_noise()
        with torch.no_grad():
            q = self.q_net.q_values(state_t)
            action = int(torch.argmax(q, dim=1).item())
        return action

    def _project_distribution(
        self,
        next_dist: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        gamma_n: float,
    ) -> torch.Tensor:
        batch_size = next_dist.size(0)
        support = self.support

        tz = rewards + (1.0 - dones) * gamma_n * support.view(1, -1)
        tz = tz.clamp(self.v_min, self.v_max)

        b = (tz - self.v_min) / self.delta_z
        l = b.floor().long()
        u = b.ceil().long()

        l = l.clamp(0, self.atoms - 1)
        u = u.clamp(0, self.atoms - 1)

        m = torch.zeros(batch_size, self.atoms, device=self.device, dtype=torch.float32)

        offset = torch.arange(batch_size, device=self.device).unsqueeze(1) * self.atoms
        m.view(-1).index_add_(
            0, (l + offset).view(-1),
            (next_dist * (u.float() - b)).view(-1)
        )
        m.view(-1).index_add_(
            0, (u + offset).view(-1),
            (next_dist * (b - l.float())).view(-1)
        )
        m = m / (m.sum(dim=1, keepdim=True) + 1e-8)
        return m

    def update(self, beta: float = 0.4) -> Optional[float]:
        if len(self.replay) < self.batch_size:
            return None

        idxs, batch, is_weights = self.replay.sample(self.batch_size, beta=beta)

        states = np.stack([tr.state for tr in batch]).astype(np.float32)
        actions = np.array([tr.action for tr in batch], dtype=np.int64)
        rewards = np.array([tr.reward for tr in batch], dtype=np.float32)
        next_states = np.stack([tr.next_state for tr in batch]).astype(np.float32)
        dones = np.array([tr.done for tr in batch], dtype=np.float32)

        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        weights_t = torch.as_tensor(is_weights, dtype=torch.float32, device=self.device)

        weights_t = weights_t / (weights_t.max() + 1e-8)

        B = states_t.size(0)
        N = self.atoms
        batch_idx = torch.arange(B, device=self.device)

        self.q_net.train()
        self.q_net.reset_noise()

        dist_all = self.q_net(states_t)
        chosen_dist = dist_all[batch_idx, actions_t].clamp(min=1e-6)
        log_p = torch.log(chosen_dist)

        with torch.no_grad():
            next_q_online = self.q_net.q_values(next_states_t)
            next_actions = next_q_online.argmax(dim=1)

            next_dist_all = self.target_net(next_states_t)
            next_dist = next_dist_all[batch_idx, next_actions].clamp(min=1e-6)

            proj_dist = self._project_distribution(
                next_dist=next_dist,
                rewards=rewards_t.view(B, 1),
                dones=dones_t.view(B, 1),
                gamma_n=self.n_gamma,
            )

        loss_per_sample = -(proj_dist * log_p).sum(dim=1)
        loss = (loss_per_sample * weights_t).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        grad_clip = getattr(self, "grad_clip_norm", None)
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), grad_clip)

        self.optimizer.step()

        new_priorities = loss_per_sample.detach().cpu().numpy() + self.replay.eps
        self.replay.update_priorities(idxs, new_priorities)


        return float(loss.item())

    def _beta_by_frame(self, frame_idx: int) -> float:
        frac = min(1.0, frame_idx / self.per_beta_frames)
        return self.per_beta_start + frac * (1.0 - self.per_beta_start)

    def learn(self, total_timesteps: int, log_name: Optional[str] = None) -> None:
        if log_name:
            log_path = os.path.join(self.log_dir, log_name)
        else:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            log_path = os.path.join(self.log_dir, f"RainbowDQN_{current_time}")

        os.makedirs(log_path, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_path)

        obs, _ = self.env.reset()
        self.nstep_acc.reset()
        ep_reward = 0.0

        for t in range(1, total_timesteps + 1):
            self.steps_done += 1

            action = self.select_action(obs, deterministic=False)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = bool(terminated or truncated)

            ep_reward += float(reward)

            one = Transition(
                state=np.array(obs, copy=False),
                action=int(action),
                reward=float(reward),
                next_state=np.array(next_obs, copy=False),
                done=done
            )

            nstep_t = self.nstep_acc.push(one)
            if nstep_t is not None:
                self.replay.push(nstep_t)

            if done:
                for tr in self.nstep_acc.flush():
                    self.replay.push(tr)

                self.episodes += 1
                if self.writer:
                    self.writer.add_scalar("rollout/ep_rew_mean", ep_reward, self.steps_done)

                print(f"Step {t}/{total_timesteps} | Episode {self.episodes} | Reward: {ep_reward:.2f}")
                obs, _ = self.env.reset()
                self.nstep_acc.reset()
                ep_reward = 0.0
            else:
                obs = next_obs

            if t % self.train_freq == 0:
                beta = self._beta_by_frame(t)
                for _ in range(self.gradient_steps):
                    loss = self.update(beta=beta)
                    if loss is not None and self.writer:
                        self.writer.add_scalar("train/loss", loss, self.steps_done)
                        self.writer.add_scalar("train/per_beta", beta, self.steps_done)

            if t % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())

        if self.writer:
            self.writer.close()
            self.writer = None

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        payload = {
            "model_state_dict": self.q_net.state_dict(),
            "atoms": self.atoms,
            "v_min": self.v_min,
            "v_max": self.v_max,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
        }
        torch.save(payload, path)

    def load(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(payload["model_state_dict"])
        self.q_net.eval()
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        action = self.select_action(obs, deterministic=deterministic)
        return action, None
