import sys
sys.path.insert(0, '../..')

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, DQN
from src.environment import ATC2DEnv
from src.agents import CustomDQN
import os
import torch

def evaluate_agent(agent, env, n_episodes=20, name="Agent"):
    rewards = []
    print(f"--- Evaluating {name} ---")
    for i in range(n_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        
        while not (done or truncated):
            if hasattr(agent, 'predict'):
                action, _ = agent.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()
            
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
        
        rewards.append(episode_reward)
    
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    print(f"{name}: Mean Reward = {mean_reward:.2f} +/- {std_reward:.2f}")
    return mean_reward, std_reward, rewards

def main():
    models_dir = "models"
    
    env = ATC2DEnv()

    results = {}

    ppo_path = os.path.join(models_dir, "ppo_atc")
    if os.path.exists(ppo_path + ".zip"):
        print("Loading PPO SB3...")
        model_ppo = PPO.load(ppo_path)
        mean, std, _ = evaluate_agent(model_ppo, env, n_episodes=50, name="PPO (SB3)")
        results["PPO (SB3)"] = mean
    else:
        print("PPO model not found. Run train.py first.")

    dqn_sb3_path = os.path.join(models_dir, "dqn_atc")
    if os.path.exists(dqn_sb3_path + ".zip"):
        print("Loading DQN SB3 (Library)...")
        model_dqn_sb3 = DQN.load(dqn_sb3_path)
        mean, std, _ = evaluate_agent(model_dqn_sb3, env, n_episodes=50, name="DQN (SB3)")
        results["DQN (Library)"] = mean
    else:
        print("DQN SB3 model not found. Run train.py first.")

    dqn_path = os.path.join(models_dir, "custom_dqn_atc.pth")
    if os.path.exists(dqn_path):
        print("Loading Custom DQN...")
        agent_dqn = CustomDQN(env, device="cpu") 
        agent_dqn.load(dqn_path)
        mean, std, _ = evaluate_agent(agent_dqn, env, n_episodes=50, name="Custom DQN")
        results["Custom DQN (Mine)"] = mean
    else:
        print("Custom DQN model not found. Run train_custom_dqn.py first.")

    if results:
        names = list(results.keys())
        values = list(results.values())
        
        plt.figure(figsize=(10, 6))
        
        colors = []
        for name in names:
            if "PPO" in name: colors.append('green')
            elif "Library" in name: colors.append('blue')
            else: colors.append('orange')

        plt.bar(names, values, color=colors)
        plt.title("Agent Performance Comparison (Final Mean Reward)")
        plt.ylabel("Mean Reward (Higher is better)")
        plt.xlabel("Agent")
        
        for i, v in enumerate(values):
            plt.text(i, v + 1, str(round(v, 2)), ha='center', fontweight='bold')

        plt.grid(axis='y', alpha=0.3)
        
        plt.savefig("rezultate_comparative.png")
        print("Graph saved as 'rezultate_comparative.png'")
    else:
        print("Could not compare agents because no saved models were found.") 


if __name__ == "__main__":
    main()
