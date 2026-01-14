"""
Comprehensive Experiment Runner for ATC RL Project
Runs multiple agents with various hyperparameters and generates comparison data
"""

import sys
sys.path.insert(0, '.')

from src.environment import ATC2DEnv
from src.agents import CustomDQN
import numpy as np
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt

# Create directories
os.makedirs("results/experiments", exist_ok=True)
os.makedirs("results/plots", exist_ok=True)
os.makedirs("logs/experiments", exist_ok=True)

class ExperimentRunner:
    """Runs comprehensive RL experiments for multiple agents"""
    
    def __init__(self, save_dir="results/experiments"):
        self.save_dir = save_dir
        self.results = {}
        
    def run_baseline_experiments(self, agent_class, agent_name, n_seeds=3, timesteps=500000):
        """Run baseline experiments with multiple random seeds"""
        print(f"\n{'='*60}")
        print(f"Running Baseline Experiments: {agent_name}")
        print(f"{'='*60}")
        
        results = []
        for seed in range(n_seeds):
            print(f"\nSeed {seed+1}/{n_seeds}")
            env = ATC2DEnv()
            
            # Set random seed
            np.random.seed(seed)
            
            # Create and train agent
            agent = agent_class(env)
            agent.learn(total_timesteps=timesteps)
            
            # Evaluate
            eval_results = self.evaluate_agent(agent, env, n_episodes=20)
            results.append({
                'seed': seed,
                'mean_reward': eval_results['mean_reward'],
                'std_reward': eval_results['std_reward'],
                'success_rate': eval_results['success_rate'],
                'safety_score': eval_results['safety_score'],
                'violations': eval_results['violations']
            })
            
            # Save model
            model_path = f"models/{agent_name}_baseline_seed{seed}.pth"
            agent.save(model_path)
            print(f"Model saved: {model_path}")
        
        # Save results
        self.results[f'{agent_name}_baseline'] = results
        self.save_results(f'{agent_name}_baseline')
        
        return results
    
    def run_hyperparameter_experiments(self, agent_class, agent_name, param_grid, timesteps=300000):
        """Run experiments with different hyperparameters"""
        print(f"\n{'='*60}")
        print(f"Running Hyperparameter Experiments: {agent_name}")
        print(f"{'='*60}")
        
        results = []
        total_experiments = sum(len(values) for values in param_grid.values())
        exp_count = 0
        
        for param_name, param_values in param_grid.items():
            for value in param_values:
                exp_count += 1
                print(f"\n[{exp_count}/{total_experiments}] Testing {param_name}={value}")
                
                env = ATC2DEnv()
                
                # Create agent with modified hyperparameter
                kwargs = {param_name: value}
                agent = agent_class(env, **kwargs)
                
                # Train
                agent.learn(total_timesteps=timesteps)
                
                # Evaluate
                eval_results = self.evaluate_agent(agent, env, n_episodes=10)
                
                results.append({
                    'parameter': param_name,
                    'value': value,
                    'mean_reward': eval_results['mean_reward'],
                    'std_reward': eval_results['std_reward'],
                    'success_rate': eval_results['success_rate'],
                    'safety_score': eval_results['safety_score'],
                    'violations': eval_results['violations']
                })
                
                # Save model
                model_path = f"models/{agent_name}_{param_name}_{value}.pth"
                agent.save(model_path)
        
        # Save results
        self.results[f'{agent_name}_hyperparams'] = results
        self.save_results(f'{agent_name}_hyperparams')
        
        return results
    
    def compare_agents(self, agents_dict, timesteps=500000, n_episodes=20):
        """Compare multiple trained agents"""
        print(f"\n{'='*60}")
        print(f"Running Agent Comparison")
        print(f"{'='*60}")
        
        results = {}
        
        for agent_name, agent_class in agents_dict.items():
            print(f"\nTraining and evaluating: {agent_name}")
            
            env = ATC2DEnv()
            agent = agent_class(env)
            
            # Train
            agent.learn(total_timesteps=timesteps)
            
            # Evaluate
            eval_results = self.evaluate_agent(agent, env, n_episodes=n_episodes)
            results[agent_name] = eval_results
            
            # Save model
            model_path = f"models/{agent_name}_comparison.pth"
            agent.save(model_path)
            
            print(f"  Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
            print(f"  Success Rate: {eval_results['success_rate']:.1%}")
            print(f"  Safety Score: {eval_results['safety_score']:.1f}")
        
        # Save comparison results
        self.results['agent_comparison'] = results
        self.save_results('agent_comparison')
        
        # Generate comparison plots
        self.plot_agent_comparison(results)
        
        return results
    
    def evaluate_agent(self, agent, env, n_episodes=10):
        """Evaluate agent performance"""
        rewards = []
        successes = 0
        violations = []
        safety_scores = []
        
        for ep in range(n_episodes):
            obs, _ = env.reset()
            done = False
            truncated = False
            episode_reward = 0
            
            while not (done or truncated):
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, done, truncated, _ = env.step(action)
                episode_reward += reward
            
            rewards.append(episode_reward)
            
            # Check success (all planes cleared)
            if env.planes_in_queue == 0 and not any(env.dep_occupied) and not env.arrival_active:
                successes += 1
            
            violations.append(env.near_misses)
            safety_scores.append(env.safety_score)
        
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'success_rate': successes / n_episodes,
            'violations': np.mean(violations),
            'safety_score': np.mean(safety_scores),
            'rewards': rewards
        }
    
    def save_results(self, experiment_name):
        """Save experiment results to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.save_dir, f"{experiment_name}_{timestamp}.json")
        
        with open(filepath, 'w') as f:
            json.dump(self.results[experiment_name], f, indent=2)
        
        print(f"\nResults saved: {filepath}")
    
    def plot_agent_comparison(self, results):
        """Generate comparison plots for multiple agents"""
        agents = list(results.keys())
        metrics = ['mean_reward', 'success_rate', 'safety_score']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, metric in enumerate(metrics):
            values = [results[agent][metric] for agent in agents]
            axes[idx].bar(agents, values, color=['#3498db', '#e74c3c', '#2ecc71'][:len(agents)])
            axes[idx].set_title(f'{metric.replace("_", " ").title()}')
            axes[idx].set_ylabel('Value')
            axes[idx].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for i, v in enumerate(values):
                axes[idx].text(i, v + 0.01 * max(values), f'{v:.2f}', 
                             ha='center', va='bottom')
        
        plt.tight_layout()
        filepath = os.path.join("results/plots", "agent_comparison.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved: {filepath}")
        plt.close()
    
    def plot_hyperparameter_analysis(self, results, agent_name):
        """Generate hyperparameter analysis plots"""
        # Group by parameter
        params = {}
        for result in results:
            param = result['parameter']
            if param not in params:
                params[param] = {'values': [], 'rewards': [], 'success': []}
            params[param]['values'].append(result['value'])
            params[param]['rewards'].append(result['mean_reward'])
            params[param]['success'].append(result['success_rate'])
        
        # Create plots
        n_params = len(params)
        fig, axes = plt.subplots(1, n_params, figsize=(6*n_params, 5))
        if n_params == 1:
            axes = [axes]
        
        for idx, (param_name, data) in enumerate(params.items()):
            axes[idx].plot(data['values'], data['rewards'], 'o-', label='Mean Reward', linewidth=2)
            axes[idx].set_xlabel(param_name)
            axes[idx].set_ylabel('Mean Reward')
            axes[idx].set_title(f'Effect of {param_name}')
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = os.path.join("results/plots", f"{agent_name}_hyperparams.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Hyperparameter plot saved: {filepath}")
        plt.close()
    
    def generate_latex_table(self, comparison_results):
        """Generate LaTeX table for documentation"""
        latex = "\\begin{table}[h]\n\\centering\n"
        latex += "\\begin{tabular}{|l|c|c|c|c|}\n\\hline\n"
        latex += "Agent & Mean Reward & Success Rate & Safety Score & Violations \\\\\n\\hline\n"
        
        for agent_name, results in comparison_results.items():
            latex += f"{agent_name} & "
            latex += f"{results['mean_reward']:.2f} ± {results['std_reward']:.2f} & "
            latex += f"{results['success_rate']:.1%} & "
            latex += f"{results['safety_score']:.1f} & "
            latex += f"{results['violations']:.2f} \\\\\n"
        
        latex += "\\hline\n\\end{tabular}\n"
        latex += "\\caption{Agent Performance Comparison}\n"
        latex += "\\label{tab:agent_comparison}\n"
        latex += "\\end{table}\n"
        
        filepath = os.path.join(self.save_dir, "comparison_table.tex")
        with open(filepath, 'w') as f:
            f.write(latex)
        
        print(f"LaTeX table saved: {filepath}")
        return latex


def main():
    """Main experiment pipeline"""
    runner = ExperimentRunner()
    
    print("="*60)
    print("ATC RL PROJECT - COMPREHENSIVE EXPERIMENTS")
    print("="*60)
    
    # Example: Run baseline with CustomDQN
    print("\n1. BASELINE EXPERIMENTS")
    baseline_results = runner.run_baseline_experiments(
        agent_class=CustomDQN,
        agent_name="CustomDQN",
        n_seeds=3,
        timesteps=500000
    )
    
    # Example: Hyperparameter tuning
    print("\n2. HYPERPARAMETER EXPERIMENTS")
    param_grid = {
        'learning_rate': [1e-4, 1e-3, 5e-3],
        'gamma': [0.95, 0.99, 0.999],
        'batch_size': [32, 64, 128]
    }
    hyperparam_results = runner.run_hyperparameter_experiments(
        agent_class=CustomDQN,
        agent_name="CustomDQN",
        param_grid=param_grid,
        timesteps=300000
    )
    runner.plot_hyperparameter_analysis(hyperparam_results, "CustomDQN")
    
    # Example: Agent comparison (add your teammates' agents here)
    print("\n3. AGENT COMPARISON")
    agents_dict = {
        'CustomDQN': CustomDQN,
        # 'PPO': PPOAgent,  # Uncomment when implemented
        # 'A3C': A3CAgent,  # Uncomment when implemented
    }
    comparison_results = runner.compare_agents(
        agents_dict=agents_dict,
        timesteps=500000,
        n_episodes=20
    )
    
    # Generate LaTeX table
    runner.generate_latex_table(comparison_results)
    
    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETED!")
    print(f"Results saved in: {runner.save_dir}")
    print(f"Plots saved in: results/plots/")
    print("="*60)


if __name__ == "__main__":
    main()
