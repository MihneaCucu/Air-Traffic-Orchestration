from stable_baselines3 import PPO, DQN
from atc_env import ATC2DEnv
import time

def test_model(algo_name, file_path):
    print(f"Testing {algo_name}...")
    
    # Aici activăm render_mode="human" ca să vedem fereastra
    env = ATC2DEnv(render_mode="human")
    
    # Încărcăm modelul
    if algo_name == "PPO":
        model = PPO.load(file_path)
    else:
        model = DQN.load(file_path)

    episodes = 5
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        score = 0
        
        while not (done or truncated):
            # Agentul prezice acțiunea
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            score += reward
            
            # Încetinit puțin vizualizarea dacă e prea rapidă
            time.sleep(0.05) 

        print(f"Episod {ep+1}: Score = {score:.2f}")
    
    env.close()

if __name__ == "__main__":
    # Decomentează linia pe care vrei să o testezi
    test_model("PPO", "models/ppo_atc")
    # test_model("DQN", "models/dqn_atc")