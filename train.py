import gymnasium as gym
from stable_baselines3 import PPO, DQN
from atc_env import ATC2DEnv
import os

# Creăm directoare pentru a salva modelele și logurile
models_dir = "models"
log_dir = "atc_logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

def train():
    # 1. Inițializăm mediul
    # Nu folosim render_mode aici pentru că vrem viteză maximă
    env = ATC2DEnv() 
    
    # --- ANTRENARE PPO ---
    print("--- Start Antrenare PPO ---")
    # În train.py
    model_ppo = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, device="cpu")
    
    # Antrenăm (poți crește timesteps la 500.000 sau 1M pentru rezultate mai bune)
    # Crește de la 100.000 la 500.000 sau 1.000.000
    model_ppo.learn(total_timesteps=1000000, tb_log_name="PPO_LongRun")
    
    # Salvăm modelul
    model_ppo.save(f"{models_dir}/ppo_atc")
    print("PPO salvat cu succes.")

    # --- ANTRENARE DQN ---
    print("--- Start Antrenare DQN ---")
    # Resetăm mediul (deși SB3 face asta intern, e bine să fim siguri)
    env.reset()
    
    model_dqn = DQN("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
    model_dqn.learn(total_timesteps=100000, tb_log_name="DQN_run")
    
    model_dqn.save(f"{models_dir}/dqn_atc")
    print("DQN salvat cu succes.")

if __name__ == "__main__":
    train()