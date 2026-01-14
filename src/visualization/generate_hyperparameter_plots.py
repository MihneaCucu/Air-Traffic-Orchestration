import matplotlib.pyplot as plt
import numpy as np
import os

OUTPUT_DIR = "documentation_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')

def plot_hyperparameter_impact():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax = axes[0, 0]
    learning_rates = [1e-4, 1e-3, 5e-3]
    lr_labels = ['1e-4', '1e-3', '5e-3']
    rewards = [35.2, 45.8, 38.5]
    stds = [6.1, 4.1, 8.3]
    
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    bars = ax.bar(range(len(learning_rates)), rewards, yerr=stds, 
                   capsize=8, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_xticks(range(len(learning_rates)))
    ax.set_xticklabels(lr_labels)
    ax.set_xlabel('Learning Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Reward', fontsize=12, fontweight='bold')
    ax.set_title('A) Impact of Learning Rate', fontsize=13, fontweight='bold', loc='left')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    for bar, val, std in zip(bars, rewards, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax = axes[0, 1]
    gammas = [0.95, 0.99, 0.995]
    gamma_labels = ['0.95', '0.99', '0.995']
    rewards = [40.2, 45.8, 44.1]
    stds = [5.5, 4.1, 4.8]
    
    bars = ax.bar(range(len(gammas)), rewards, yerr=stds,
                   capsize=8, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_xticks(range(len(gammas)))
    ax.set_xticklabels(gamma_labels)
    ax.set_xlabel('Gamma (Discount Factor)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Reward', fontsize=12, fontweight='bold')
    ax.set_title('B) Impact of Discount Factor', fontsize=13, fontweight='bold', loc='left')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    for bar, val, std in zip(bars, rewards, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax = axes[1, 0]
    batch_sizes = [32, 64, 128]
    batch_labels = ['32', '64', '128']
    rewards = [42.3, 45.8, 43.7]
    stds = [5.2, 4.1, 4.9]
    
    bars = ax.bar(range(len(batch_sizes)), rewards, yerr=stds,
                   capsize=8, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_xticks(range(len(batch_sizes)))
    ax.set_xticklabels(batch_labels)
    ax.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Reward', fontsize=12, fontweight='bold')
    ax.set_title('C) Impact of Batch Size', fontsize=13, fontweight='bold', loc='left')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    for bar, val, std in zip(bars, rewards, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax = axes[1, 1]
    
    config_names = ['baseline', 'high_lr', 'low_lr', 'high_gamma', 
                    'low_gamma', 'large_batch', 'small_batch']
    short_names = ['Base', 'HighLR', 'LowLR', 'HighG', 'LowG', 'LrgB', 'SmlB']
    config_rewards = [45.8, 38.5, 35.2, 44.1, 40.2, 43.7, 42.3]
    
    sorted_idx = np.argsort(config_rewards)[::-1]
    sorted_names = [short_names[i] for i in sorted_idx]
    sorted_rewards = [config_rewards[i] for i in sorted_idx]
    
    colors_gradient = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(sorted_rewards)))
    bars = ax.barh(range(len(sorted_names)), sorted_rewards, 
                    color=colors_gradient, edgecolor='black', linewidth=1.5)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=10)
    ax.set_xlabel('Mean Reward', fontsize=12, fontweight='bold')
    ax.set_title('D) Configuration Ranking', fontsize=13, fontweight='bold', loc='left')
    ax.grid(axis='x', alpha=0.3)
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    
    for bar, val in zip(bars, sorted_rewards):
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2.,
                f'{val:.1f}', ha='left', va='center', fontsize=9, fontweight='bold')
    
    plt.suptitle('Hyperparameter Sensitivity Analysis', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = f"{OUTPUT_DIR}/hyperparameter_impact.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_exploration_vs_exploitation():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    steps = np.arange(0, 100000, 100)
    
    epsilon_995 = np.maximum(0.01, 1.0 * (0.995 ** (steps / 100)))
    epsilon_998 = np.maximum(0.01, 1.0 * (0.998 ** (steps / 100)))
    epsilon_990 = np.maximum(0.01, 1.0 * (0.990 ** (steps / 100)))
    
    ax1.plot(steps, epsilon_995, label='decay=0.995 (baseline)', 
             linewidth=2.5, color='#2E86AB')
    ax1.plot(steps, epsilon_998, label='decay=0.998 (slower)', 
             linewidth=2.5, color='#A23B72')
    ax1.plot(steps, epsilon_990, label='decay=0.990 (faster)', 
             linewidth=2.5, color='#F18F01')
    
    ax1.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Epsilon Value', fontsize=12, fontweight='bold')
    ax1.set_title('A) Epsilon Decay Strategies', fontsize=13, fontweight='bold', loc='left')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    
    decay_rates = ['0.990\n(fast)', '0.995\n(baseline)', '0.998\n(slow)']
    rewards = [41.2, 45.8, 43.5]
    stds = [6.3, 4.1, 5.0]
    colors = ['#F18F01', '#2E86AB', '#A23B72']
    
    bars = ax2.bar(range(len(decay_rates)), rewards, yerr=stds,
                    capsize=8, color=colors, alpha=0.8, 
                    edgecolor='black', linewidth=1.5)
    ax2.set_xticks(range(len(decay_rates)))
    ax2.set_xticklabels(decay_rates, fontsize=10)
    ax2.set_xlabel('Epsilon Decay Rate', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mean Reward', fontsize=12, fontweight='bold')
    ax2.set_title('B) Impact on Final Performance', fontsize=13, fontweight='bold', loc='left')
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    for bar, val, std in zip(bars, rewards, stds):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + std + 1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.suptitle('Exploration-Exploitation Trade-off Analysis', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = f"{OUTPUT_DIR}/exploration_exploitation.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def create_results_table_image():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    data = [
        ['Configuration', 'LR', 'Gamma', 'Batch', 'Mean Reward', 'Std Dev', 'Time (h)'],
        ['Baseline', '1e-3', '0.99', '64', '45.8', '4.1', '1.8'],
        ['High LR', '5e-3', '0.99', '64', '38.5', '8.3', '1.7'],
        ['Low LR', '1e-4', '0.99', '64', '35.2', '6.1', '2.1'],
        ['High Gamma', '1e-3', '0.995', '64', '44.1', '4.8', '1.9'],
        ['Low Gamma', '1e-3', '0.95', '64', '40.2', '5.5', '1.7'],
        ['Large Batch', '1e-3', '0.99', '128', '43.7', '4.9', '1.9'],
        ['Small Batch', '1e-3', '0.99', '32', '42.3', '5.2', '1.8'],
    ]
    
    cell_colors = []
    for i, row in enumerate(data):
        if i == 0: 
            cell_colors.append(['#2E86AB'] * len(row))
        else:
            reward = float(row[4])
            if reward > 44:
                color = '#d4edda'
            elif reward > 40:
                color = '#fff3cd'
            else:
                color = '#f8d7da' 
            cell_colors.append([color] * len(row))
    
    table = ax.table(cellText=data, cellLoc='center', loc='center',
                     cellColours=cell_colors, colWidths=[0.18, 0.12, 0.12, 0.12, 0.15, 0.12, 0.12])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    for i in range(len(data[0])):
        cell = table[(0, i)]
        cell.set_text_props(weight='bold', color='white', fontsize=11)
    
    for i in range(1, len(data)):
        cell = table[(i, 0)]
        cell.set_text_props(weight='bold', fontsize=10)
    
    plt.title('Hyperparameter Experiment Results Summary', 
              fontsize=16, fontweight='bold', pad=20)
    
    output_path = f"{OUTPUT_DIR}/results_table.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def main():
    plot_hyperparameter_impact()
    plot_exploration_vs_exploitation()
    create_results_table_image()
    
    print("\n" + "="*70)
    print("HYPERPARAMETER PLOTS GENERATED!")
    print("="*70)
    print(f"\nOutput directory: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    print("  1. hyperparameter_impact.png - Impact of each hyperparameter")
    print("  2. exploration_exploitation.png - Epsilon decay analysis")
    print("  3. results_table.png - Visual results table")
    
    print("\nNote: These plots use example data.")
    print("   After running experiments, update with real results from:")
    print("   - experiment_results/all_experiments_summary.json")
    print("   - Or use analyze_experiments.py to generate real plots")
    print("="*70)


if __name__ == "__main__":
    main()
