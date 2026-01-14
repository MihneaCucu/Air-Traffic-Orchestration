import gymnasium as gym
from atc_env import ATC2DEnv
from custom_dqn_agent import CustomDQN
import os
import json
import numpy as np
from datetime import datetime

models_dir = "models/experiments"
log_dir = "atc_logs"
results_dir = "experiment_results"

for dir_path in [models_dir, log_dir, results_dir]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

EXPERIMENTS = [
    {
        "name": "baseline",
        "learning_rate": 1e-3,
        "gamma": 0.99,
        "batch_size": 64,
        "buffer_size": 50000,
        "target_update_freq": 1000,
        "epsilon_decay": 0.995,
    },
    {
        "name": "high_lr",
        "learning_rate": 5e-3,
        "gamma": 0.99,
        "batch_size": 64,
        "buffer_size": 50000,
        "target_update_freq": 1000,
        "epsilon_decay": 0.995,
    },
    {
        "name": "low_lr",
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "batch_size": 64,
        "buffer_size": 50000,
        "target_update_freq": 1000,
        "epsilon_decay": 0.995,
    },
    {
        "name": "high_gamma",
        "learning_rate": 1e-3,
        "gamma": 0.995,
        "batch_size": 64,
        "buffer_size": 50000,
        "target_update_freq": 1000,
        "epsilon_decay": 0.995,
    },
    {
        "name": "low_gamma",
        "learning_rate": 1e-3,
        "gamma": 0.95,
        "batch_size": 64,
        "buffer_size": 50000,
        "target_update_freq": 1000,
        "epsilon_decay": 0.995,
    },
    {
        "name": "large_batch",
        "learning_rate": 1e-3,
        "gamma": 0.99,
        "batch_size": 128,
        "buffer_size": 50000,
        "target_update_freq": 1000,
        "epsilon_decay": 0.995,
    },
    {
        "name": "small_batch",
        "learning_rate": 1e-3,
        "gamma": 0.99,
        "batch_size": 32,
        "buffer_size": 50000,
        "target_update_freq": 1000,
        "epsilon_decay": 0.995,
    },
    {
        "name": "freq_target_update",
        "learning_rate": 1e-3,
        "gamma": 0.99,
        "batch_size": 64,
        "buffer_size": 50000,
        "target_update_freq": 500,
        "epsilon_decay": 0.995,
    },
    {
        "name": "slow_epsilon_decay",
        "learning_rate": 1e-3,
        "gamma": 0.99,
        "batch_size": 64,
        "buffer_size": 50000,
        "target_update_freq": 1000,
        "epsilon_decay": 0.998,
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
    
    agent = CustomDQN(
        env,
        learning_rate=config['learning_rate'],
        gamma=config['gamma'],
        batch_size=config['batch_size'],
        buffer_size=config['buffer_size'],
        target_update_freq=config['target_update_freq'],
        epsilon_decay=config.get('epsilon_decay', 0.995),
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
    
    print(f"\n✓ Experiment completed!")
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
    print(f"# Starting Hyperparameter Experiments")
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
                print(f"❌ Error in experiment {config['name']}_seed{seed}: {e}")
                continue
    
    summary_file = f"{results_dir}/all_experiments_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    print(f"{'Configuration':<25} {'Mean Reward':<15} {'Std Dev':<15}")
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
        print(f"{config_name:<25} {mean_reward:>12.2f}    {std_reward:>12.2f}")
    
    print(f"{'='*70}")
    print(f"\n✓ All experiments completed!")
    print(f"  Results saved to: {results_dir}/")
    print(f"  Summary file: {summary_file}")
    print(f"\nNext steps:")
    print(f"  1. Run: python analyze_experiments.py")
    print(f"  2. View TensorBoard logs: tensorboard --logdir {log_dir}")
    
    return all_results


if __name__ == "__main__":
    results = run_all_experiments()
