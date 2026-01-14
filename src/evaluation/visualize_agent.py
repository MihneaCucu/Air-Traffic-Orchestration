import gymnasium as gym
import gymnasium as gym
from stable_baselines3 import PPO
from atc_env import ATC2DEnv
from custom_dqn_agent import CustomDQN
import os

AGENT_TYPE = "CustomDQN" 

env = ATC2DEnv(render_mode="human")

model = None
try:
    if AGENT_TYPE == "PPO":
        model_path = "models/ppo_atc"
        print(f"Încărcare PPO din {model_path}...")
        model = PPO.load(model_path)
    elif AGENT_TYPE == "CustomDQN":
        model_path = "models/custom_dqn_atc.pth"
        print(f"Încărcare Custom DQN din {model_path}...")
        agent = CustomDQN(env, device="cpu")
        agent.load(model_path)
        model = agent
except Exception as e:
    print(f"Eroare la încărcarea modelului: {e}")
    exit()

episodes = 5
for ep in range(episodes):
    obs, _ = env.reset()
    done = False
    truncated = False
    score = 0
    
    print(f"--- Episodul {ep + 1} ---")
    
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        
        obs, reward, done, truncated, info = env.step(action)
        score += reward
        
    print(f"Scor final: {score}")

env.close()