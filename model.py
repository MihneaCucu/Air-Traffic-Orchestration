from stable_baselines3 import PPO
from atc_env import ATC2DEnv

env = ATC2DEnv() # Fără render_mode pentru viteză
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

model.save("ppo_atc")