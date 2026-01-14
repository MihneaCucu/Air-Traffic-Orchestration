import gymnasium as gym
from atc_env import ATC2DEnv
from custom_ppo_agent import CustomPPO
import os

models_dir = "models"
log_dir = "atc_logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

def train():
    env = ATC2DEnv()
    agent = CustomPPO(
        env,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        log_dir=log_dir
    )
    
    agent.learn(total_timesteps=500000)
    
    save_path = f"{models_dir}/custom_ppo_atc.pth"
    agent.save(save_path)
    print(f"Custom PPO salvat cu succes la {save_path}")

if __name__ == "__main__":
    train()
