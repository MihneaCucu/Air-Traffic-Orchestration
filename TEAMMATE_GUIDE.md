# 👥 Teammate Implementation Guide

## 🎯 Your Mission

Implement **at least 2 more RL agents** (total 3+ required) that work with our ATC environment.

## 📋 Requirements Checklist

- [ ] Agent uses `ATC2DEnv` from `src/environment`
- [ ] Agent follows similar structure to `CustomDQN`
- [ ] Agent has `.learn()` and `.predict()` methods
- [ ] Agent can be saved/loaded with `.save()` and `.load()`
- [ ] Agent trains successfully (scores improving over time)
- [ ] Agent is added to comparison experiments

---

## 🤖 Suggested Agents to Implement

### Option 1: **PPO** (Policy Gradient - EASY)
Use Stable-Baselines3 (already in requirements):

```python
# src/agents/ppo_agent.py
from stable_baselines3 import PPO
from src.environment import ATC2DEnv

class PPOAgent:
    def __init__(self, env, learning_rate=3e-4, **kwargs):
        self.env = env
        self.model = PPO("MlpPolicy", env, 
                        learning_rate=learning_rate,
                        verbose=1, **kwargs)
    
    def learn(self, total_timesteps=500000):
        self.model.learn(total_timesteps=total_timesteps)
    
    def predict(self, obs, deterministic=True):
        return self.model.predict(obs, deterministic=deterministic)
    
    def save(self, path):
        self.model.save(path)
    
    @classmethod
    def load(cls, path, env):
        agent = cls(env)
        agent.model = PPO.load(path)
        return agent
```

### Option 2: **SARSA** (Tabular Q-Learning - MEDIUM)
Classic on-policy algorithm:

```python
# src/agents/sarsa_agent.py
import numpy as np
from collections import defaultdict

class SARSAAgent:
    def __init__(self, env, learning_rate=0.1, gamma=0.99, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.env = env
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))
    
    def discretize_state(self, obs):
        """Convert continuous state to discrete for Q-table"""
        # Simple binning strategy
        return tuple(np.round(obs, 1))
    
    def predict(self, obs, deterministic=False):
        state = self.discretize_state(obs)
        
        if not deterministic and np.random.rand() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.q_table[state])
        
        return action, None
    
    def learn(self, total_timesteps=500000):
        obs, _ = self.env.reset()
        state = self.discretize_state(obs)
        action, _ = self.predict(obs)
        
        for step in range(total_timesteps):
            # Take action
            next_obs, reward, done, truncated, _ = self.env.step(action)
            next_state = self.discretize_state(next_obs)
            
            # Choose next action
            next_action, _ = self.predict(next_obs)
            
            # SARSA update
            current_q = self.q_table[state][action]
            next_q = self.q_table[next_state][next_action]
            new_q = current_q + self.lr * (reward + self.gamma * next_q - current_q)
            self.q_table[state][action] = new_q
            
            # Transition
            state = next_state
            action = next_action
            obs = next_obs
            
            if done or truncated:
                obs, _ = self.env.reset()
                state = self.discretize_state(obs)
                action, _ = self.predict(obs)
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            if step % 10000 == 0:
                print(f"Step {step}/{total_timesteps}, Epsilon: {self.epsilon:.3f}")
    
    def save(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'q_table': dict(self.q_table),
                'epsilon': self.epsilon,
                'params': {
                    'lr': self.lr,
                    'gamma': self.gamma,
                    'epsilon_decay': self.epsilon_decay,
                    'epsilon_min': self.epsilon_min
                }
            }, f)
    
    @classmethod
    def load(cls, path, env):
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        agent = cls(env, **data['params'])
        agent.q_table = defaultdict(lambda: np.zeros(env.action_space.n), data['q_table'])
        agent.epsilon = data['epsilon']
        return agent
```

### Option 3: **A2C** (Actor-Critic - EASY with SB3)

```python
# src/agents/a2c_agent.py
from stable_baselines3 import A2C

class A2CAgent:
    def __init__(self, env, learning_rate=7e-4, **kwargs):
        self.env = env
        self.model = A2C("MlpPolicy", env, 
                        learning_rate=learning_rate,
                        verbose=1, **kwargs)
    
    def learn(self, total_timesteps=500000):
        self.model.learn(total_timesteps=total_timesteps)
    
    def predict(self, obs, deterministic=True):
        return self.model.predict(obs, deterministic=deterministic)
    
    def save(self, path):
        self.model.save(path)
    
    @classmethod
    def load(cls, path, env):
        agent = cls(env)
        agent.model = A2C.load(path)
        return agent
```

---

## 🔧 Implementation Steps

### Step 1: Create Your Agent File
```bash
# Create new file in src/agents/
touch src/agents/your_agent_name.py
```

### Step 2: Implement the Agent
Copy one of the templates above and modify as needed.

### Step 3: Update `__init__.py`
```python
# src/agents/__init__.py
from .custom_dqn_agent import CustomDQN
from .your_agent_name import YourAgent  # Add this line

__all__ = ['CustomDQN', 'YourAgent']  # Add your agent here
```

### Step 4: Test Your Agent
```python
# test_my_agent.py
from src.environment import ATC2DEnv
from src.agents import YourAgent

env = ATC2DEnv()
agent = YourAgent(env)

print("Training for 10k steps...")
agent.learn(total_timesteps=10000)

print("Testing...")
obs, _ = env.reset()
for _ in range(100):
    action, _ = agent.predict(obs, deterministic=True)
    obs, reward, done, truncated, _ = env.step(action)
    if done or truncated:
        break

print("✅ Agent works!")
agent.save(f"models/{YourAgent.__name__}_test.pth")
```

### Step 5: Add to Comparison Experiments
```python
# In run_comprehensive_experiments.py, line ~260
agents_dict = {
    'CustomDQN': CustomDQN,
    'YourAgentName': YourAgent,  # Add this line
}
```

---

## 📊 Running Experiments

### After all agents are implemented:

```bash
# 1. Run comprehensive experiments
python run_comprehensive_experiments.py

# 2. Generate plots
python src/visualization/generate_all_plots.py

# 3. Compare agents visually
python src/evaluation/compare_agents.py
```

---

## 🎯 Expected Results Format

Your agent should produce results like:

```
Episode 1: 260.4
Episode 2: 297.4
Episode 3: 264.8
Episode 4: 278.4
Episode 5: 259.1

Average: 272.0 ± 15.8
Success Rate: 80%
Safety Score: 95/100
Violations: 0.4 per episode
```

---

## ⚠️ Common Issues & Solutions

### Issue 1: Import Errors
**Problem**: `ModuleNotFoundError: No module named 'src'`

**Solution**: 
```python
import sys
sys.path.insert(0, '.')
from src.environment import ATC2DEnv
```

### Issue 2: Environment Shape Mismatch
**Problem**: `Expected input of shape X, got Y`

**Solution**: Our environment has 9-dimensional state space. Check:
```python
print(env.observation_space.shape)  # Should be (9,)
print(env.action_space.n)  # Should be 3
```

### Issue 3: Poor Performance
**Problem**: Agent not learning (scores staying negative)

**Solutions**:
- Train longer (500k+ timesteps)
- Tune learning rate (try 1e-4, 1e-3, 1e-2)
- Check epsilon decay (for ε-greedy agents)
- Verify reward signals are working

### Issue 4: Agent Too Slow
**Problem**: Training takes forever

**Solutions**:
- Don't use `render_mode="human"` during training
- Use smaller batch sizes for faster updates
- Reduce total_timesteps for initial testing

---

## ✅ Definition of Done

Your agent is ready when:

1. ✅ Code is in `src/agents/your_agent.py`
2. ✅ Agent trains without errors
3. ✅ Agent can be saved and loaded
4. ✅ Scores are improving over training
5. ✅ Agent added to `__init__.py`
6. ✅ Agent tested with visualization
7. ✅ Agent added to comparison experiments

---

## 🤝 Communication

**Before implementing:**
- Tell the team which agent you're working on (avoid duplicates)
- Ask questions if the API is unclear

**After implementing:**
- Share your results (average scores)
- Commit your code
- Update this document with your agent details

---

## 📞 Need Help?

**Environment questions**: Check `src/environment/atc_env.py`
**DQN reference**: See `src/agents/custom_dqn_agent.py`
**Training issues**: Check `src/training/train_custom_dqn.py`

**Still stuck?** Ask the team! We're all learning together 🚀

---

## 🎓 Algorithm Diversity (Required for Full Points)

Make sure your team implements:
- ✅ **Value-based**: DQN (already done)
- 🔄 **Policy-based**: PPO or A2C or REINFORCE
- 🔄 **Tabular or Different**: SARSA or Q-Learning or Monte Carlo

This ensures you get full points for "algorithm diversity"!

---

Good luck! 🍀
