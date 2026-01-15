import sys
sys.path.insert(0, '..')

from src.environment import ATC2DEnv
from ppo_agent import PPOAgent
import os

models_dir = "../models"
log_dir = "../logs/ppo"

os.makedirs(models_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

def train():
    print("="*60)
    print("Training PPO Agent (From Scratch)")
    print("="*60)
    env = ATC2DEnv()
    agent = PPOAgent(
        env,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        update_epochs=4,
        device="cpu"
    )
    agent.learn(total_timesteps=500000, n_steps=2048, log_interval=10)
    save_path = f"{models_dir}/ppo_scratch.pth"
    agent.save(save_path)
    print(f"\n Training completed!")
    print(f"Model saved to: {save_path}")

if __name__ == "__main__":
    train()
