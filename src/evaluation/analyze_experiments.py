import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

results_dir = "experiment_results"

def load_all_results():
    results = []
    
    if not os.path.exists(results_dir):
        print(f"Error: {results_dir} directory not found!")
        return results
    
    for file in Path(results_dir).glob("*_results.json"):
        with open(file, 'r') as f:
            results.append(json.load(f))
    
    return results


def group_by_config(results):
    grouped = {}
    
    for result in results:
        config_name = result['config']['name']
        if config_name not in grouped:
            grouped[config_name] = []
        grouped[config_name].append(result)
    
    return grouped


def print_summary_table(grouped_results):
    print("\n" + "="*100)
    print("HYPERPARAMETER EXPERIMENT RESULTS")
    print("="*100)
    print(f"{'Config':<20} {'LR':<10} {'Gamma':<8} {'Batch':<8} {'Mean ±Std':<20} {'Min':<10} {'Max':<10}")
    print("-"*100)
    
    summary_data = []
    
    for config_name, results in sorted(grouped_results.items()):
        rewards = [r['evaluation']['mean_reward'] for r in results]
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        min_reward = np.min(rewards)
        max_reward = np.max(rewards)
        
        config = results[0]['config']
        lr = config['learning_rate']
        gamma = config['gamma']
        batch_size = config['batch_size']
        
        print(f"{config_name:<20} {lr:<10.0e} {gamma:<8.3f} {batch_size:<8} "
              f"{mean_reward:>7.2f} ±{std_reward:<5.2f}   {min_reward:>7.2f}    {max_reward:>7.2f}")
        
        summary_data.append({
            'name': config_name,
            'mean': mean_reward,
            'std': std_reward,
            'min': min_reward,
            'max': max_reward,
            'config': config
        })
    
    print("="*100)
    
    best_config = max(summary_data, key=lambda x: x['mean'])
    print(f"\nBEST CONFIGURATION: {best_config['name']}")
    print(f"   Mean Reward: {best_config['mean']:.2f} ± {best_config['std']:.2f}")
    print(f"   Learning Rate: {best_config['config']['learning_rate']}")
    print(f"   Gamma: {best_config['config']['gamma']}")
    print(f"   Batch Size: {best_config['config']['batch_size']}")
    
    return summary_data


def plot_comparison(grouped_results, save_path="experiment_results/comparison.png"):
    config_names = []
    means = []
    stds = []
    
    for config_name, results in sorted(grouped_results.items()):
        rewards = [r['evaluation']['mean_reward'] for r in results]
        config_names.append(config_name)
        means.append(np.mean(rewards))
        stds.append(np.std(rewards))
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Hyperparameter Experiment Results', fontsize=16, fontweight='bold')
    
    ax1 = axes[0, 0]
    x_pos = np.arange(len(config_names))
    ax1.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Configuration', fontweight='bold')
    ax1.set_ylabel('Mean Reward', fontweight='bold')
    ax1.set_title('Mean Reward by Configuration (with std dev)')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(config_names, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    ax2 = axes[0, 1]
    all_rewards = []
    for config_name in sorted(grouped_results.keys()):
        rewards = [r['evaluation']['mean_reward'] for r in grouped_results[config_name]]
        all_rewards.append(rewards)
    
    bp = ax2.boxplot(all_rewards, labels=config_names, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_edgecolor('black')
    ax2.set_xlabel('Configuration', fontweight='bold')
    ax2.set_ylabel('Reward Distribution', fontweight='bold')
    ax2.set_title('Reward Distribution Across Seeds')
    ax2.set_xticklabels(config_names, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    ax3 = axes[1, 0]
    lr_configs = {}
    for config_name, results in grouped_results.items():
        lr = results[0]['config']['learning_rate']
        rewards = [r['evaluation']['mean_reward'] for r in results]
        lr_configs[lr] = lr_configs.get(lr, []) + rewards
    
    if len(lr_configs) > 1:
        lrs = sorted(lr_configs.keys())
        lr_means = [np.mean(lr_configs[lr]) for lr in lrs]
        lr_stds = [np.std(lr_configs[lr]) for lr in lrs]
        
        ax3.errorbar(lrs, lr_means, yerr=lr_stds, marker='o', capsize=5, 
                     linewidth=2, markersize=8, color='darkgreen')
        ax3.set_xlabel('Learning Rate', fontweight='bold')
        ax3.set_ylabel('Mean Reward', fontweight='bold')
        ax3.set_title('Impact of Learning Rate')
        ax3.set_xscale('log')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    else:
        ax3.text(0.5, 0.5, 'Not enough LR variations', ha='center', va='center', transform=ax3.transAxes)
    
    ax4 = axes[1, 1]
    gamma_configs = {}
    for config_name, results in grouped_results.items():
        gamma = results[0]['config']['gamma']
        rewards = [r['evaluation']['mean_reward'] for r in results]
        gamma_configs[gamma] = gamma_configs.get(gamma, []) + rewards
    
    if len(gamma_configs) > 1:
        gammas = sorted(gamma_configs.keys())
        gamma_means = [np.mean(gamma_configs[g]) for g in gammas]
        gamma_stds = [np.std(gamma_configs[g]) for g in gammas]
        
        ax4.errorbar(gammas, gamma_means, yerr=gamma_stds, marker='s', capsize=5,
                     linewidth=2, markersize=8, color='darkviolet')
        ax4.set_xlabel('Gamma (Discount Factor)', fontweight='bold')
        ax4.set_ylabel('Mean Reward', fontweight='bold')
        ax4.set_title('Impact of Discount Factor')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    else:
        ax4.text(0.5, 0.5, 'Not enough gamma variations', ha='center', va='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Comparison plot saved to: {save_path}")
    plt.show()


def generate_report(summary_data, save_path="experiment_results/report.txt"):
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("RAPORT EXPERIMENTE - CUSTOM DQN\n")
        f.write("="*80 + "\n\n")
        
        f.write("1. CONFIGURAȚII TESTATE\n")
        f.write("-"*80 + "\n\n")
        
        for item in summary_data:
            f.write(f"Configurație: {item['name']}\n")
            f.write(f"  - Learning Rate: {item['config']['learning_rate']}\n")
            f.write(f"  - Gamma: {item['config']['gamma']}\n")
            f.write(f"  - Batch Size: {item['config']['batch_size']}\n")
            f.write(f"  - Buffer Size: {item['config']['buffer_size']}\n")
            f.write(f"  - Target Update Frequency: {item['config']['target_update_freq']}\n")
            f.write(f"  - Rezultat: {item['mean']:.2f} ± {item['std']:.2f}\n\n")
        
        f.write("\n2. ANALIZA REZULTATELOR\n")
        f.write("-"*80 + "\n\n")
        
        best = max(summary_data, key=lambda x: x['mean'])
        worst = min(summary_data, key=lambda x: x['mean'])
        
        f.write(f"Cea mai bună configurație: {best['name']}\n")
        f.write(f"  - Reward mediu: {best['mean']:.2f} ± {best['std']:.2f}\n")
        f.write(f"  - Parametri: LR={best['config']['learning_rate']}, "
                f"Gamma={best['config']['gamma']}, Batch={best['config']['batch_size']}\n\n")
        
        f.write(f"Cea mai slabă configurație: {worst['name']}\n")
        f.write(f"  - Reward mediu: {worst['mean']:.2f} ± {worst['std']:.2f}\n\n")
        
        f.write("3. CONCLUZII\n")
        f.write("-"*80 + "\n\n")
        f.write("- Experimentele au fost realizate cu 3 seed-uri diferite pentru fiecare configurație\n")
        f.write("- Variabilele testate: learning rate, gamma, batch size, epsilon decay\n")
        f.write(f"- Diferența între cea mai bună și cea mai slabă: {best['mean'] - worst['mean']:.2f}\n")
        f.write("- Stabilitatea antrenamentului (std dev) variază între configurații\n\n")
        
        f.write("="*80 + "\n")
    
    print(f"Report saved to: {save_path}")


def main():
    results = load_all_results()
    
    if not results:
        print("No results found! Please run experiments first with: python run_experiments.py")
        return
    
    print(f"Found {len(results)} experiment results")
    
    grouped_results = group_by_config(results)
    print(f"Configurations tested: {len(grouped_results)}")
    
    summary_data = print_summary_table(grouped_results)
    
    plot_comparison(grouped_results)
    
    generate_report(summary_data)
    
    print("\nAnalysis complete!")
    print("\nGenerated files:")
    print("  - experiment_results/comparison.png")
    print("  - experiment_results/report.txt")


if __name__ == "__main__":
    main()
