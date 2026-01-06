import gymnasium as gym
from atc_env import ATC2DEnv
from custom_dqn_agent import CustomDQN
import os

# Create directories
models_dir = "models"
log_dir = "atc_logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

def train():
    print("--- Start Antrenare Custom DQN ---")
    
    # Initialize environment
    env = ATC2DEnv() 
    
    # Initialize Custom DQN Agent
    # We can tune parameters here
    agent = CustomDQN(
        env, 
        learning_rate=1e-3, 
        gamma=0.99, 
        batch_size=64, 
        buffer_size=50000, 
        target_update_freq=1000,
        log_dir=log_dir
    )
    
    # Train
    # Increase timesteps for better results (e.g. 500.000 or 1.000.000)
    agent.learn(total_timesteps=1000000)
    
    # Save model
    save_path = f"{models_dir}/custom_dqn_atc.pth"
    agent.save(save_path)
    print(f"Custom DQN salvat cu succes la {save_path}")

if __name__ == "__main__":
    train()
