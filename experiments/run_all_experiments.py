import sys
sys.path.insert(0, '.')

from src.environment import ATC2DEnv
import numpy as np
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import time

os.makedirs("experiments/results", exist_ok=True)
os.makedirs("experiments/plots", exist_ok=True)
os.makedirs("experiments/logs", exist_ok=True)
os.makedirs("experiments/models", exist_ok=True)

sns.set_style("whitegrid")

class ComprehensiveExperiments:
    def __init__(self):
        self.results = defaultdict(dict)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    def import_agents(self):
        agents = {}
        try:
            sys.path.insert(0, 'RandomAgent')
            from random_agent import RandomAgent
            agents['Random'] = RandomAgent
            print(" Random agent loaded (baseline)")
        except Exception as e:
            print(f"  Random agent not found: {e}")
        try:
            sys.path.insert(0, 'DQN')
            from custom_dqn_agent import CustomDQN
            agents['DQN'] = CustomDQN
            print(" DQN agent loaded")
        except Exception as e:
            print(f"  DQN agent not found: {e}")
        try:
            sys.path.insert(0, 'PPO')
            from ppo_agent import PPOAgent
            agents['PPO'] = PPOAgent
            print(" PPO agent loaded")
        except Exception as e:
            print(f"  PPO agent not found: {e}")
        try:
            sys.path.insert(0, 'A2C')
            from agent_a2c import A2CAgent
            agents['A2C'] = A2CAgent
            print(" A2C agent loaded")
        except Exception as e:
            print(f"  A2C agent not found: {e}")
        try:
            sys.path.insert(0, 'SAC')
            from sac_agent import DiscreteSAC
            agents['SAC'] = DiscreteSAC
            print(" SAC agent loaded")
        except Exception as e:
            print(f"  SAC agent not found: {e}")
        return agents
    def evaluate_agent(self, agent, env, n_episodes=20, seed=None):
        metrics = {
            'rewards': [],
            'successes': [],
            'violations': [],
            'safety_scores': [],
            'episode_lengths': [],
            'fuel_emergencies': []
        }
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=seed)
            done = False
            truncated = False
            episode_reward = 0
            steps = 0
            fuel_critical_count = 0
            while not (done or truncated):
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, done, truncated, _ = env.step(action)
                episode_reward += reward
                steps += 1
                if hasattr(env, 'fuel_timers'):
                    fuel_critical_count += sum(1 for f in env.fuel_timers if f < 5)
            metrics['rewards'].append(episode_reward)
            metrics['episode_lengths'].append(steps)
            metrics['violations'].append(env.near_misses)
            metrics['safety_scores'].append(env.safety_score)
            metrics['fuel_emergencies'].append(fuel_critical_count)
            success = (env.planes_in_queue == 0 and 
                      not any(env.dep_occupied) and 
                      not env.arrival_active)
            metrics['successes'].append(1 if success else 0)
        return {
            'mean_reward': np.mean(metrics['rewards']),
            'std_reward': np.std(metrics['rewards']),
            'median_reward': np.median(metrics['rewards']),
            'min_reward': np.min(metrics['rewards']),
            'max_reward': np.max(metrics['rewards']),
            'success_rate': np.mean(metrics['successes']),
            'mean_violations': np.mean(metrics['violations']),
            'mean_safety_score': np.mean(metrics['safety_scores']),
            'mean_episode_length': np.mean(metrics['episode_lengths']),
            'convergence_stability': np.std(metrics['rewards'][-5:]) if len(metrics['rewards']) >= 5 else np.std(metrics['rewards']),
            'fuel_emergency_rate': np.mean(metrics['fuel_emergencies']) / np.mean(metrics['episode_lengths']),
            'raw_rewards': metrics['rewards']
        }
    def experiment_1_baseline(self, agents, n_seeds=5, timesteps=500000, eval_episodes=30):
        print("\n" + "="*80)
        print("EXPERIMENT 1: BASELINE PERFORMANCE (Multiple Seeds)")
        print("="*80)
        print(f"Seeds: {n_seeds} | Training: {timesteps} steps | Evaluation: {eval_episodes} episodes")
        print("="*80 + "\n")
        agent_timesteps = {
            'Random': 0,
            'PPO': timesteps // 5,
            'DQN': timesteps,
            'A2C': timesteps,
            'SAC': timesteps
        }
        for agent_name, AgentClass in agents.items():
            print(f"\n Testing {agent_name}...")
            seed_results = []
            for seed in range(n_seeds):
                print(f"  Seed {seed + 1}/{n_seeds}...")
                model_path = f"experiments/models/{agent_name}_baseline_seed{seed}.pth"
                env = ATC2DEnv()
                np.random.seed(seed)
                if agent_name == 'SAC':
                    state_dim = env.observation_space.shape[0]
                    action_dim = env.action_space.n
                    agent = AgentClass(state_dim, action_dim)
                else:
                    agent = AgentClass(env)
                if os.path.exists(model_path):
                    print(f"     Model exists, loading instead of retraining...")
                    agent.load(model_path)
                    training_time = 0
                else:
                    start_time = time.time()
                    train_timesteps = agent_timesteps.get(agent_name, timesteps)
                    if train_timesteps > 0:
                        if agent_name == 'SAC':
                            agent.learn(total_timesteps=train_timesteps, env=env)
                        else:
                            agent.learn(total_timesteps=train_timesteps)
                    training_time = time.time() - start_time
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    agent.save(model_path)
                eval_results = self.evaluate_agent(agent, env, n_episodes=eval_episodes, seed=seed)
                eval_results['training_time'] = training_time
                eval_results['seed'] = seed
                seed_results.append(eval_results)
                print(f"    Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
            self.results['baseline'][agent_name] = seed_results
        self._save_results('experiment1_baseline')
        self._plot_baseline_comparison()
    def experiment_2_hyperparameters(self, agents, base_timesteps=50000):
        print("\n" + "="*80)
        print("EXPERIMENT 2: HYPERPARAMETER SENSITIVITY")
        print("="*80 + "\n")
        param_grids = {
            'DQN': {
                'learning_rate': [1e-4, 5e-4, 1e-3, 3e-3],
                'gamma': [0.90, 0.95, 0.99, 0.995],
                'batch_size': [32, 64, 128],
                'target_update_freq': [500, 1000, 2000],
                'epsilon_decay': [0.995, 0.997, 0.999]
            },
            'PPO': {
                'learning_rate': [1e-4, 3e-4, 1e-3, 3e-3],
                'clip_epsilon': [0.1, 0.2, 0.3],
                'entropy_coef': [0.0, 0.01, 0.05],
                'gamma': [0.90, 0.95, 0.99]
            },
            'A2C': {
                'learning_rate': [1e-4, 5e-4, 1e-3, 3e-3],
                'gamma': [0.90, 0.95, 0.99, 0.995],
                'entropy_coef': [0.01, 0.1, 0.2]
            },
            'SAC': {
                'lr': [1e-4, 3e-4, 1e-3, 3e-3],
                'alpha': [0.05, 0.1, 0.2, 0.5],
                'tau': [0.001, 0.005, 0.01]
            }
        }
        for agent_name, AgentClass in agents.items():
            if agent_name not in param_grids:
                continue
            print(f"\n Testing {agent_name} hyperparameters...")
            agent_hyperparam_results = []
            for param_name, param_values in param_grids[agent_name].items():
                for value in param_values:
                    model_path = f"experiments/models/{agent_name}_hyperparam_{param_name}_{value}.pth"
                    env = ATC2DEnv()
                    kwargs = {param_name: value}
                    if agent_name == 'SAC':
                        state_dim = env.observation_space.shape[0]
                        action_dim = env.action_space.n
                        agent = AgentClass(state_dim, action_dim, **kwargs)
                    else:
                        agent = AgentClass(env, **kwargs)
                    if os.path.exists(model_path):
                        print(f"  {param_name}={value}...  Model exists, loading...")
                        agent.load(model_path)
                    else:
                        print(f"  {param_name}={value}... Training...")
                        if agent_name == 'SAC':
                            agent.learn(total_timesteps=base_timesteps, env=env)
                        else:
                            agent.learn(total_timesteps=base_timesteps)
                        agent.save(model_path)
                    eval_results = self.evaluate_agent(agent, env, n_episodes=15)
                    agent_hyperparam_results.append({
                        'parameter': param_name,
                        'value': value,
                        **eval_results
                    })
            self.results['hyperparameters'][agent_name] = agent_hyperparam_results
        self._save_results('experiment2_hyperparameters')
        self._plot_hyperparameter_effects()
    def experiment_3_convergence(self, agents, eval_intervals=8, total_steps=100000):
        print("\n" + "="*80)
        print("EXPERIMENT 3: CONVERGENCE & STABILITY ANALYSIS")
        print("="*80 + "\n")
        for agent_name, AgentClass in agents.items():
            print(f"\n Analyzing {agent_name} convergence...")
            env = ATC2DEnv()
            if agent_name == 'SAC':
                state_dim = env.observation_space.shape[0]
                action_dim = env.action_space.n
                agent = AgentClass(state_dim, action_dim)
            else:
                agent = AgentClass(env)
            convergence_data = []
            steps_per_eval = total_steps // eval_intervals
            for i in range(eval_intervals):
                current_steps = (i + 1) * steps_per_eval
                if agent_name == 'SAC':
                    agent.learn(total_timesteps=steps_per_eval, env=env)
                else:
                    agent.learn(total_timesteps=steps_per_eval)
                eval_results = self.evaluate_agent(agent, env, n_episodes=10)
                convergence_data.append({
                    'timestep': current_steps,
                    **eval_results
                })
                print(f"  Step {current_steps}: Reward {eval_results['mean_reward']:.2f}")
            self.results['convergence'][agent_name] = convergence_data
        self._save_results('experiment3_convergence')
        self._plot_convergence_curves()
    def experiment_4_failure_analysis(self, agents, n_episodes=50):
        print("\n" + "="*80)
        print("EXPERIMENT 4: FAILURE MODE ANALYSIS")
        print("="*80 + "\n")
        for agent_name, AgentClass in agents.items():
            print(f"\n🔍 Analyzing {agent_name} failures...")
            env = ATC2DEnv()
            agent = AgentClass(env)
            model_path = f"models/{agent_name.lower()}_*.pth"
            failure_modes = {
                'collision_failures': 0,
                'fuel_failures': 0,
                'timeout_failures': 0,
                'total_episodes': n_episodes
            }
            for ep in range(n_episodes):
                obs, _ = env.reset()
                done = False
                truncated = False
                while not (done or truncated):
                    action, _ = agent.predict(obs, deterministic=True)
                    obs, reward, done, truncated, _ = env.step(action)
                if env.near_misses > 5:
                    failure_modes['collision_failures'] += 1
                if any(f < 0 for f in env.fuel_timers if hasattr(env, 'fuel_timers')):
                    failure_modes['fuel_failures'] += 1
                if truncated and not done:
                    failure_modes['timeout_failures'] += 1
            self.results['failures'][agent_name] = failure_modes
            print(f"  Collision failures: {failure_modes['collision_failures']}")
            print(f"  Fuel failures: {failure_modes['fuel_failures']}")
            print(f"  Timeout failures: {failure_modes['timeout_failures']}")
        self._save_results('experiment4_failures')
        self._plot_failure_modes()
    def _save_results(self, experiment_name):
        filepath = f"experiments/results/{experiment_name}_{self.timestamp}.json"
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        cleaned_results = json.loads(json.dumps(self.results, default=convert))
        with open(filepath, 'w') as f:
            json.dump(cleaned_results, f, indent=2)
        print(f"\n💾 Results saved: {filepath}")
    def _plot_baseline_comparison(self):
        if 'baseline' not in self.results:
            return
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Baseline Performance Comparison Across All Agents', fontsize=16, fontweight='bold')
        agents = list(self.results['baseline'].keys())
        colors = ['
        ax = axes[0, 0]
        means = [np.mean([s['mean_reward'] for s in self.results['baseline'][agent]]) for agent in agents]
        stds = [np.std([s['mean_reward'] for s in self.results['baseline'][agent]]) for agent in agents]
        bars = ax.bar(agents, means, yerr=stds, capsize=5, alpha=0.7, color=colors)
        ax.set_ylabel('Mean Reward', fontsize=12)
        ax.set_title('Average Reward (Multiple Seeds)', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(i, m + s + 10, f'{m:.1f}±{s:.1f}', ha='center', va='bottom')
        ax = axes[0, 1]
        success_rates = [np.mean([s['success_rate'] for s in self.results['baseline'][agent]]) * 100 for agent in agents]
        ax.bar(agents, success_rates, alpha=0.7, color=colors)
        ax.set_ylabel('Success Rate (%)', fontsize=12)
        ax.set_title('Episode Completion Success Rate', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)
        ax = axes[1, 0]
        safety = [np.mean([s['mean_safety_score'] for s in self.results['baseline'][agent]]) for agent in agents]
        ax.bar(agents, safety, alpha=0.7, color=colors)
        ax.set_ylabel('Safety Score', fontsize=12)
        ax.set_title('Average Safety Score', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)
        ax = axes[1, 1]
        variances = [np.mean([s['convergence_stability'] for s in self.results['baseline'][agent]]) for agent in agents]
        ax.bar(agents, variances, alpha=0.7, color=colors)
        ax.set_ylabel('Reward Std Dev (Lower is Better)', fontsize=12)
        ax.set_title('Performance Stability', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        filepath = f"experiments/plots/baseline_comparison_{self.timestamp}.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f" Plot saved: {filepath}")
        plt.close()
    def _plot_hyperparameter_effects(self):
        pass
    def _plot_convergence_curves(self):
        if 'convergence' not in self.results:
            return
        plt.figure(figsize=(12, 6))
        for agent_name, data in self.results['convergence'].items():
            steps = [d['timestep'] for d in data]
            rewards = [d['mean_reward'] for d in data]
            plt.plot(steps, rewards, marker='o', label=agent_name, linewidth=2)
        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel('Mean Reward', fontsize=12)
        plt.title('Convergence Curves - All Agents', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        filepath = f"experiments/plots/convergence_{self.timestamp}.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f" Plot saved: {filepath}")
        plt.close()
    def _plot_failure_modes(self):
        if 'failures' not in self.results:
            return
        agents = list(self.results['failures'].keys())
        failure_types = ['collision_failures', 'fuel_failures', 'timeout_failures']
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(agents))
        width = 0.25
        for i, failure_type in enumerate(failure_types):
            counts = [self.results['failures'][agent][failure_type] for agent in agents]
            ax.bar(x + i * width, counts, width, label=failure_type.replace('_', ' ').title())
        ax.set_xlabel('Agents', fontsize=12)
        ax.set_ylabel('Failure Count', fontsize=12)
        ax.set_title('Failure Mode Analysis', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(agents)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        filepath = f"experiments/plots/failures_{self.timestamp}.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f" Plot saved: {filepath}")
        plt.close()
    def generate_latex_report(self):
        latex = []
        if 'baseline' in self.results:
            latex.append("\\begin{table}[h]\n\\centering")
            latex.append("\\begin{tabular}{|l|c|c|c|c|}")
            latex.append("\\hline")
            latex.append("Agent & Mean Reward & Success Rate & Safety Score & Stability \\\\\n\\hline")
            for agent_name in self.results['baseline'].keys():
                data = self.results['baseline'][agent_name]
                mean_r = np.mean([s['mean_reward'] for s in data])
                std_r = np.std([s['mean_reward'] for s in data])
                success = np.mean([s['success_rate'] for s in data]) * 100
                safety = np.mean([s['mean_safety_score'] for s in data])
                stability = np.mean([s['convergence_stability'] for s in data])
                latex.append(f"{agent_name} & {mean_r:.1f}$\\pm${std_r:.1f} & {success:.1f}\\% & {safety:.1f} & {stability:.2f} \\\\")
            latex.append("\\hline\n\\end{tabular}")
            latex.append("\\caption{Baseline Performance Comparison}")
            latex.append("\\label{tab:baseline}")
            latex.append("\\end{table}\n\n")
        latex_str = "\n".join(latex)
        filepath = f"experiments/results/report_{self.timestamp}.tex"
        with open(filepath, 'w') as f:
            f.write(latex_str)
        print(f" LaTeX report saved: {filepath}")
        return latex_str


def main():
    print("="*80)
    print("COMPREHENSIVE MULTI-AGENT EXPERIMENT SUITE")
    print("ATC Reinforcement Learning Project")
    print("="*80 + "\n")
    experiments = ComprehensiveExperiments()
    print("🔍 Loading agents...")
    agents = experiments.import_agents()
    if not agents:
        print("\n No agents found! Make sure agent implementations exist in their folders.")
        return
    print(f"\n Found {len(agents)} agents: {list(agents.keys())}\n")
    try:
        experiments.experiment_1_baseline(agents, n_seeds=3, timesteps=500000, eval_episodes=30)
        experiments.experiment_2_hyperparameters(agents, base_timesteps=300000)
        experiments.experiment_3_convergence(agents, eval_intervals=10, total_steps=500000)
        experiments.experiment_4_failure_analysis(agents, n_episodes=50)
        experiments.generate_latex_report()
        print("\n" + "="*80)
        print(" ALL EXPERIMENTS COMPLETED!")
        print("="*80)
        print(f"\n📁 Results: experiments/results/")
        print(f" Plots: experiments/plots/")
        print("\nYou now have comprehensive data for:")
        print("   Baseline performance (multiple seeds)")
        print("   Hyperparameter sensitivity")
        print("   Convergence analysis")
        print("   Failure mode analysis")
        print("   Statistical significance")
        print("   LaTeX tables for documentation")
    except Exception as e:
        print(f"\n Error during experiments: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
