import gymnasium as gym
from stable_baselines3 import PPO
from atc_env import ATC2DEnv

# 1. Încărcăm mediul cu render_mode='human' ca să vedem fereastra
env = ATC2DEnv(render_mode="human")

# 2. Încărcăm modelul antrenat
# Asigură-te că fișierul ppo_atc.zip este în același folder
model = PPO.load("models/ppo_atc")

# 3. Rulăm câteva episoade de test
episodes = 5
for ep in range(episodes):
    obs, _ = env.reset()
    done = False
    truncated = False
    score = 0
    
    print(f"--- Episodul {ep + 1} ---")
    
    while not (done or truncated):
        # Modelul prezice acțiunea. 
        # deterministic=True înseamnă că alege cea mai probabilă acțiune (nu mai explorează)
        action, _ = model.predict(obs, deterministic=True)
        
        obs, reward, done, truncated, info = env.step(action)
        score += reward
        
        # O mică pauză opțională dacă se mișcă prea repede, 
        # deși env.render() are deja limitare la 30 FPS
        # import time; time.sleep(0.05)

    print(f"Scor final: {score}")

env.close()