from atc_env import ATC2DEnv
import time

# Inițializăm mediul cu render_mode activat
env = ATC2DEnv(render_mode="human")
obs, _ = env.reset()

print("Start ATC Simulation...")

for _ in range(500):
    # Alegem o acțiune random (0-4)
    # Aici vei pune mai târziu modelul tău (ex: model.predict(obs))
    action = env.action_space.sample() 
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        print(f"Episod terminat! Reward final: {reward}")
        obs, _ = env.reset()
        
env.close()