import sys
sys.path.insert(0, '..')

from src.environment import ATC2DEnv
from random_agent import RandomAgent
import numpy as np


def evaluate_random_agent(n_episodes=20):
    print("="*80)
    print("RANDOM AGENT EVALUATION")
    print("="*80)
    print("\nRandom agent selects actions uniformly at random.")
    print("No training required - provides baseline for comparison.\n")
    env = ATC2DEnv()
    agent = RandomAgent(env)
    rewards = []
    print(f"Evaluating for {n_episodes} episodes...\n")
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        steps = 0
        while not (done or truncated):
            action, _ = agent.predict(obs)
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            steps += 1
        rewards.append(episode_reward)
        print(f"Episode {ep+1:2d}: Reward = {episode_reward:7.2f}, Steps = {steps:3d}")
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Mean Reward:   {np.mean(rewards):7.2f}")
    print(f"Std Reward:    {np.std(rewards):7.2f}")
    print(f"Min Reward:    {np.min(rewards):7.2f}")
    print(f"Max Reward:    {np.max(rewards):7.2f}")
    print(f"Median Reward: {np.median(rewards):7.2f}")
    print(f"{'='*80}\n")
    print(" This baseline shows what performance looks like WITHOUT learning.")
    print("   Trained agents should significantly outperform this!\n")


if __name__ == "__main__":
    evaluate_random_agent(n_episodes=30)
