import gymnasium as gym
from atc_env import ATC2DEnv
from agent_a2c import A2CAgent
import os
import json
import numpy as np
from datetime import datetime
import time

models_dir = "models"
log_dir = "atc_logs"
results_dir = "experiment_results"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

def evaluate(agent, env, episodes=20, seed=None):
    """Evaluate agent performance"""
    rewards = []
    
    for _ in range(episodes):
        state, _ = env.reset(seed=seed)
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state, deterministic=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        rewards.append(episode_reward)
    
    return np.array(rewards)

def train(seed=42, total_timesteps=25000, config_name="A2C"):
    """Train A2C agent with given seed"""
    print(f"\n{'='*60}")
    print(f"Training A2C Agent - {config_name} - Seed {seed}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    env = ATC2DEnv()
    
    agent = A2CAgent(
        env,
        learning_rate=3e-4,
        gamma=0.99,
        entropy_coef=0.1,
        value_coef=0.5,
        log_dir=log_dir,
        seed=seed
    )
    
    # Train
    log_name = f"A2C_seed{seed}"
    agent.learn(total_timesteps=total_timesteps, n_steps=20, log_name=log_name)
    
    training_time = time.time() - start_time
    
    # Evaluate
    print(f"\nEvaluating A2C Agent ({config_name}, seed {seed})...")
    eval_rewards = evaluate(agent, env, episodes=20, seed=seed)
    mean_reward = np.mean(eval_rewards)
    std_reward = np.std(eval_rewards)
    min_reward = np.min(eval_rewards)
    max_reward = np.max(eval_rewards)
    
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    
    # Save model
    model_path = f"{models_dir}/a2c_seed{seed}.pth"
    agent.save(model_path)
    print(f"Model saved: {model_path}")
    
    # Save results as JSON (format colegilor)
    results = {
        "experiment_name": f"{config_name}_seed{seed}",
        "config": {
            "name": config_name,
            "learning_rate": agent.learning_rate,
            "gamma": agent.gamma,
            "entropy_coef": agent.entropy_coef,
            "value_coef": agent.value_coef,
        },
        "seed": seed,
        "training_time_seconds": training_time,
        "total_timesteps": total_timesteps,
        "evaluation": {
            "mean_reward": float(mean_reward),
            "std_reward": float(std_reward),
            "min_reward": float(min_reward),
            "max_reward": float(max_reward)
        },
        "timestamp": datetime.now().isoformat()
    }
    
    results_path = f"{results_dir}/{config_name}_seed{seed}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {results_path}")
    
    env.close()
    return mean_reward, std_reward

def main():
    """Run training with multiple seeds"""
    seeds = [0, 1, 2]
    all_results = []
    
    for seed in seeds:
        mean_reward, std_reward = train(seed=seed, config_name="A2C")
        all_results.append({
            "seed": seed,
            "mean_reward": mean_reward,
            "std_reward": std_reward
        })
    
    # Print summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY - A2C Agent")
    print(f"{'='*60}")
    
    for result in all_results:
        print(f"Seed {result['seed']}: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
    
    overall_mean = np.mean([r["mean_reward"] for r in all_results])
    overall_std = np.std([r["mean_reward"] for r in all_results])
    print(f"\nOverall: {overall_mean:.2f} ± {overall_std:.2f}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
