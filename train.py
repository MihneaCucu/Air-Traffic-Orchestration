import gymnasium as gym
from stable_baselines3 import PPO, DQN
from atc_env import ATC2DEnv
import os

models_dir = "models"
log_dir = "atc_logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

def train():
    env = ATC2DEnv() 
    
    print("--- Start Antrenare PPO ---")
    model_ppo = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, device="cpu")
    
    model_ppo.learn(total_timesteps=1000000, tb_log_name="PPO_LongRun")
    
    model_ppo.save(f"{models_dir}/ppo_atc")
    print("PPO salvat cu succes.")

    print("--- Start Antrenare DQN ---")
    env.reset()
    
    model_dqn = DQN("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
    model_dqn.learn(total_timesteps=100000, tb_log_name="DQN_run")
    
    model_dqn.save(f"{models_dir}/dqn_atc")
    print("DQN salvat cu succes.")

if __name__ == "__main__":
    train()