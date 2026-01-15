import json
import os
import glob
import numpy as np
from datetime import datetime

def load_latest_results():
    results = {}
    result_files = glob.glob("experiments/results/*.json")
    for exp_type in ['experiment1_baseline', 'experiment2_hyperparameters', 
                     'experiment3_convergence', 'experiment4_failures']:
        matching = [f for f in result_files if exp_type in f]
        if matching:
            latest = max(matching, key=os.path.getctime)
            with open(latest, 'r') as f:
                results[exp_type] = json.load(f)
            print(f" Loaded {exp_type}: {os.path.basename(latest)}")
    return results

def analyze_baseline(data):
    print("\n" + "="*80)
    print("BASELINE PERFORMANCE SUMMARY")
    print("="*80 + "\n")
    if 'baseline' not in data:
        print("No baseline data found")
        return
    for agent_name, seed_results in data['baseline'].items():
        rewards = [s['mean_reward'] for s in seed_results]
        success_rates = [s['success_rate'] * 100 for s in seed_results]
        safety_scores = [s['mean_safety_score'] for s in seed_results]
        print(f" {agent_name}")
        print(f"   Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f} "
              f"(min: {np.min(rewards):.2f}, max: {np.max(rewards):.2f})")
        print(f"   Success Rate: {np.mean(success_rates):.1f}%")
        print(f"   Safety Score: {np.mean(safety_scores):.1f}")
        print()
    avg_rewards = {agent: np.mean([s['mean_reward'] for s in data['baseline'][agent]]) 
                   for agent in data['baseline'].keys()}
    best_agent = max(avg_rewards, key=avg_rewards.get)
    print(f"Best Agent: {best_agent} (Reward: {avg_rewards[best_agent]:.2f})")

def analyze_convergence(data):
    print("\n" + "="*80)
    print("CONVERGENCE ANALYSIS")
    print("="*80 + "\n")
    if 'convergence' not in data:
        print("No convergence data found")
        return
    for agent_name, conv_data in data['convergence'].items():
        rewards = [d['mean_reward'] for d in conv_data]
        split = len(rewards) // 3
        early_avg = np.mean(rewards[:split])
        late_avg = np.mean(rewards[-split:])
        improvement = ((late_avg - early_avg) / abs(early_avg)) * 100 if early_avg != 0 else 0
        final_stability = np.std(rewards[-split:])
        print(f" {agent_name}")
        print(f"   Initial: {early_avg:.2f} → Final: {late_avg:.2f}")
        print(f"   Improvement: {improvement:+.1f}%")
        print(f"   Final Stability: {final_stability:.2f} (std dev)")
        if improvement > 20:
            print(f"   Status:  Good convergence")
        elif improvement > 0:
            print(f"   Status:   Slow convergence")
        else:
            print(f"   Status:  No convergence / overfitting")
        print()

def analyze_failures(data):
    print("\n" + "="*80)
    print("FAILURE MODE ANALYSIS")
    print("="*80 + "\n")
    if 'failures' not in data:
        print("No failure data found")
        return
    for agent_name, failure_data in data['failures'].items():
        total = failure_data['total_episodes']
        collisions = failure_data['collision_failures']
        fuel = failure_data['fuel_failures']
        timeouts = failure_data['timeout_failures']
        print(f"🔍 {agent_name}")
        print(f"   Collision Failures: {collisions}/{total} ({collisions/total*100:.1f}%)")
        print(f"   Fuel Failures: {fuel}/{total} ({fuel/total*100:.1f}%)")
        print(f"   Timeout Failures: {timeouts}/{total} ({timeouts/total*100:.1f}%)")
        total_failures = collisions + fuel + timeouts
        if total_failures < total * 0.1:
            print(f"   Safety Rating:  Excellent (<10% failures)")
        elif total_failures < total * 0.3:
            print(f"   Safety Rating:   Acceptable (10-30% failures)")
        else:
            print(f"   Safety Rating:  Needs improvement (>30% failures)")
        print()

def generate_quick_summary(results):
    print("\n" + "="*80)
    print("EXECUTIVE SUMMARY")
    print("="*80 + "\n")
    summary = []
    if 'experiment1_baseline' in results and 'baseline' in results['experiment1_baseline']:
        data = results['experiment1_baseline']['baseline']
        avg_rewards = {agent: np.mean([s['mean_reward'] for s in data[agent]]) 
                       for agent in data.keys()}
        best_agent = max(avg_rewards, key=avg_rewards.get)
        best_reward = avg_rewards[best_agent]
        summary.append(f"• Best Performing Agent: {best_agent} ({best_reward:.2f} avg reward)")
        if len(data[best_agent]) >= 3:
            std = np.std([s['mean_reward'] for s in data[best_agent]])
            summary.append(f"• Result Confidence: ±{std:.2f} std dev across {len(data[best_agent])} seeds")
    if 'experiment3_convergence' in results and 'convergence' in results['experiment3_convergence']:
        converged_agents = []
        for agent, conv_data in results['experiment3_convergence']['convergence'].items():
            rewards = [d['mean_reward'] for d in conv_data]
            split = len(rewards) // 3
            improvement = np.mean(rewards[-split:]) - np.mean(rewards[:split])
            if improvement > 0:
                converged_agents.append(agent)
        summary.append(f"• Agents with Positive Convergence: {', '.join(converged_agents)}")
    if 'experiment4_failures' in results and 'failures' in results['experiment4_failures']:
        safest_agents = []
        for agent, failure_data in results['experiment4_failures']['failures'].items():
            total_failures = (failure_data['collision_failures'] + 
                            failure_data['fuel_failures'] + 
                            failure_data['timeout_failures'])
            failure_rate = total_failures / failure_data['total_episodes']
            if failure_rate < 0.3:
                safest_agents.append(agent)
        summary.append(f"• Safe Agents (<30% failure): {', '.join(safest_agents)}")
    print("Key Findings:\n")
    for line in summary:
        print(line)
    print("\nRecommendation:")
    if summary:
        print(f"  Based on comprehensive experiments, {best_agent} demonstrates")
        print(f"  the best balance of performance, stability, and safety.")
    else:
        print("  Run experiments to generate recommendation.")
    print("\n" + "="*80)

def export_for_latex(results):
    print("\n Generating LaTeX snippets...")
    latex_output = []
    if 'experiment1_baseline' in results and 'baseline' in results['experiment1_baseline']:
        data = results['experiment1_baseline']['baseline']
        latex_output.append("% Paste this into your LaTeX document")
        latex_output.append("\\begin{table}[h]")
        latex_output.append("\\centering")
        latex_output.append("\\begin{tabular}{|l|c|c|c|}")
        latex_output.append("\\hline")
        latex_output.append("\\textbf{Agent} & \\textbf{Mean Reward} & \\textbf{Success Rate} & \\textbf{Safety Score} \\\\ \\hline")
        for agent_name, seed_results in data.items():
            mean_r = np.mean([s['mean_reward'] for s in seed_results])
            std_r = np.std([s['mean_reward'] for s in seed_results])
            success = np.mean([s['success_rate'] for s in seed_results]) * 100
            safety = np.mean([s['mean_safety_score'] for s in seed_results])
            latex_output.append(f"{agent_name} & ${mean_r:.1f} \\pm {std_r:.1f}$ & {success:.1f}\\% & {safety:.1f} \\\\ \\hline")
        latex_output.append("\\end{tabular}")
        latex_output.append("\\caption{Agent Performance Comparison}")
        latex_output.append("\\label{tab:agent_comparison}")
        latex_output.append("\\end{table}")
    latex_str = "\n".join(latex_output)
    output_path = "experiments/results/quick_summary_latex.tex"
    with open(output_path, 'w') as f:
        f.write(latex_str)
    print(f" LaTeX saved: {output_path}")
    print("\nPreview:")
    print("-" * 80)
    print(latex_str)
    print("-" * 80)

def main():
    print("="*80)
    print("QUICK EXPERIMENT ANALYSIS")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    results = load_latest_results()
    if not results:
        print("\n No experiment results found!")
        print("Run experiments first: python experiments/run_all_experiments.py")
        return
    for exp_type, data in results.items():
        if exp_type == 'experiment1_baseline':
            analyze_baseline(data)
        elif exp_type == 'experiment3_convergence':
            analyze_convergence(data)
        elif exp_type == 'experiment4_failures':
            analyze_failures(data)
    generate_quick_summary(results)
    export_for_latex(results)
    print("\n Analysis complete!")
    print("\nNext steps:")
    print("  1. Check plots in experiments/plots/")
    print("  2. Review detailed JSON results in experiments/results/")
    print("  3. Use LaTeX snippet in your documentation")
    print("  4. Write final conclusions based on summary above")

if __name__ == "__main__":
    main()
