import sys
sys.path.insert(0, '../..')

import gymnasium as gym
from src.environment import ATC2DEnv
from src.agents.custom_ppo_agent import CustomPPO
import os
import json
import numpy as np
from datetime import datetime

models_dir = "models/experiments_ppo"
log_dir = "atc_logs"
results_dir = "experiment_results"

for dir_path in [models_dir, log_dir, results_dir]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

EXPERIMENTS = [
    {
        "name": "PPO_baseline",
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
    },
    {
        "name": "PPO_high_lr",
        "learning_rate": 1e-3,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
    },
    {
        "name": "PPO_low_lr",
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
    },
    {
        "name": "PPO_high_gamma",
        "learning_rate": 3e-4,
        "gamma": 0.995,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
    },
    {
        "name": "PPO_low_gamma",
        "learning_rate": 3e-4,
        "gamma": 0.95,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
    },
    {
        "name": "PPO_large_batch",
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "n_steps": 2048,
        "batch_size": 128,
        "n_epochs": 10,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
    },
    {
        "name": "PPO_small_batch",
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "n_steps": 2048,
        "batch_size": 32,
        "n_epochs": 10,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
    },
    {
        "name": "PPO_high_clip",
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.3,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
    },
    {
        "name": "PPO_low_clip",
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.1,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
    },
    {
        "name": "PPO_more_epochs",
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 20,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
    },
    {
        "name": "PPO_few_epochs",
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 5,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
    },
    {
        "name": "PPO_high_entropy",
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "ent_coef": 0.05,
        "vf_coef": 0.5,
    },
    {
        "name": "PPO_low_entropy",
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "ent_coef": 0.001,
        "vf_coef": 0.5,
    },
]

NUM_SEEDS = 3
TOTAL_TIMESTEPS = 300000


def evaluate_agent(agent, env, n_episodes=10):
    total_rewards = []
    
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        
        while not (done or truncated):
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
        
        total_rewards.append(episode_reward)
    
    return {
        "mean_reward": np.mean(total_rewards),
        "std_reward": np.std(total_rewards),
        "min_reward": np.min(total_rewards),
        "max_reward": np.max(total_rewards),
    }


def run_single_experiment(config, seed, exp_idx, total_experiments):
    exp_name = f"{config['name']}_seed{seed}"
    print(f"\n{'='*70}")
    print(f"Experiment {exp_idx}/{total_experiments}: {exp_name}")
    print(f"{'='*70}")
    print(f"Configuration:")
    for key, value in config.items():
        if key != 'name':
            print(f"  {key}: {value}")
    print(f"{'='*70}\n")
    
    env = ATC2DEnv()
    env.reset(seed=seed)
    
    agent = CustomPPO(
        env,
        learning_rate=config['learning_rate'],
        gamma=config['gamma'],
        gae_lambda=config.get('gae_lambda', 0.95),
        clip_range=config.get('clip_range', 0.2),
        n_steps=config.get('n_steps', 2048),
        batch_size=config['batch_size'],
        n_epochs=config.get('n_epochs', 10),
        ent_coef=config.get('ent_coef', 0.01),
        vf_coef=config.get('vf_coef', 0.5),
        log_dir=log_dir,
        seed=seed
    )
    
    start_time = datetime.now()
    agent.learn(total_timesteps=TOTAL_TIMESTEPS, log_name=exp_name)
    training_time = (datetime.now() - start_time).total_seconds()
    
    eval_env = ATC2DEnv()
    eval_results = evaluate_agent(agent, eval_env, n_episodes=20)
    
    model_path = f"{models_dir}/{exp_name}.pth"
    agent.save(model_path)
    
    results = {
        "experiment_name": exp_name,
        "config": config,
        "seed": seed,
        "training_time_seconds": training_time,
        "total_timesteps": TOTAL_TIMESTEPS,
        "evaluation": eval_results,
        "timestamp": datetime.now().isoformat(),
    }
    
    result_file = f"{results_dir}/{exp_name}_results.json"
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n Experiment completed!")
    print(f"  Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"  Training Time: {training_time:.1f}s")
    print(f"  Results saved to: {result_file}")
    
    env.close()
    eval_env.close()
    
    return results


def run_all_experiments():
    all_results = []
    total_experiments = len(EXPERIMENTS) * NUM_SEEDS
    exp_counter = 0
    
    print(f"\n{'#'*70}")
    print(f"# Starting PPO Hyperparameter Experiments")
    print(f"# Total experiments to run: {total_experiments}")
    print(f"# Configurations: {len(EXPERIMENTS)}")
    print(f"# Seeds per configuration: {NUM_SEEDS}")
    print(f"# Total timesteps per experiment: {TOTAL_TIMESTEPS}")
    print(f"{'#'*70}\n")
    
    for config in EXPERIMENTS:
        for seed in range(NUM_SEEDS):
            exp_counter += 1
            try:
                result = run_single_experiment(config, seed, exp_counter, total_experiments)
                all_results.append(result)
            except Exception as e:
                print(f"Error in experiment {config['name']}_seed{seed}: {e}")
                continue
    
    summary_file = f"{results_dir}/all_ppo_experiments_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("PPO EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    print(f"{'Configuration':<30} {'Mean Reward':<15} {'Std Dev':<15}")
    print(f"{'-'*70}")
    
    config_results = {}
    for result in all_results:
        config_name = result['config']['name']
        if config_name not in config_results:
            config_results[config_name] = []
        config_results[config_name].append(result['evaluation']['mean_reward'])
    
    for config_name, rewards in sorted(config_results.items()):
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        print(f"{config_name:<30} {mean_reward:>12.2f}    {std_reward:>12.2f}")
    
    print(f"{'='*70}")
    print(f"\n✓ All PPO experiments completed!")
    print(f"  Results saved to: {results_dir}/")
    print(f"  Summary file: {summary_file}")
    print(f"\nNext steps:")
    print(f"  1. Run: python analyze_experiments.py")
    print(f"  2. View TensorBoard logs: tensorboard --logdir {log_dir}")
    
    return all_results


if __name__ == "__main__":
    results = run_all_experiments()
