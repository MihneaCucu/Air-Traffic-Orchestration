import sys
sys.path.insert(0, '../..')

from stable_baselines3 import PPO
from src.environment import ATC2DEnv

env = ATC2DEnv() 
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=500000)

model.save("ppo_atc")