import gymnasium as gym
from atc_env import ATC2DEnv
from agent_a2c import A2CAgent
import numpy as np
import os

SEED = 2 
MODEL_PATH = f"models/a2c_seed{SEED}.pth"
EPISODES = 5
RENDER = True

print(f"\n{'='*60}")
print(f"A2C Agent Visualization")
print(f"{'='*60}")
print(f"Model: {MODEL_PATH}")
print(f"Seed: {SEED}")
print(f"Episodes: {EPISODES}")
print(f"Render: {RENDER}\n")

# Initialize environment
render_mode = "human" if RENDER else None
env = ATC2DEnv(render_mode=render_mode)

# Initialize and load agent
try:
    agent = A2CAgent(env, seed=SEED, device="cpu")
    agent.load(MODEL_PATH)
    print(f"✓ Model loaded from {MODEL_PATH}\n")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    exit(1)

# Run episodes
all_rewards = []

for ep in range(EPISODES):
    obs, _ = env.reset()
    done = False
    episode_reward = 0
    steps = 0
    
    print(f"--- Episode {ep + 1} / {EPISODES} ---")
    
    while not done:
        action = agent.select_action(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        steps += 1
    
    all_rewards.append(episode_reward)
    print(f"Episode Reward: {episode_reward:.2f} | Steps: {steps}")

mean_reward = np.mean(all_rewards)
std_reward = np.std(all_rewards)
min_reward = np.min(all_rewards)
max_reward = np.max(all_rewards)

print(f"\n{'='*60}")
print("EVALUATION SUMMARY")
print(f"{'='*60}")
print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
print(f"Min Reward:  {min_reward:.2f}")
print(f"Max Reward:  {max_reward:.2f}")
print(f"Episodes:    {EPISODES}")
print(f"{'='*60}\n")

env.close()
