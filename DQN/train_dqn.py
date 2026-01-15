import sys
sys.path.insert(0, '..')

import gymnasium as gym
from src.environment import ATC2DEnv
from custom_dqn_agent import CustomDQN
import os

models_dir = "../models"
log_dir = "../logs/dqn"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

def train():
    env = ATC2DEnv() 
    agent = CustomDQN(
        env, 
        learning_rate=1e-3, 
        gamma=0.99, 
        batch_size=64, 
        buffer_size=50000, 
        target_update_freq=1000,
        log_dir=log_dir
    )
    agent.learn(total_timesteps=500000)
    save_path = f"{models_dir}/dqn_custom.pth"
    agent.save(save_path)
    print(f"Custom DQN salvat cu succes la {save_path}")

if __name__ == "__main__":
    train()
