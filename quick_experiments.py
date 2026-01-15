import os
import sys
import argparse
from datetime import datetime

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def run_baseline_experiments():
    print("\n" + "="*70)
    print("RUNNING BASELINE EXPERIMENTS")
    print("="*70)
    from experiments.run_all_experiments import ComprehensiveExperiments
    runner = ComprehensiveExperiments()
    agents = runner.import_agents()
    runner.experiment_1_baseline(
        agents=agents,
        timesteps=100000,
        n_seeds=3,
        eval_episodes=15
    )
    print("\n Baseline experiments complete!")


def run_hyperparameter_experiments():
    print("\n" + "="*70)
    print("RUNNING HYPERPARAMETER EXPERIMENTS")
    print("="*70)
    from experiments.run_all_experiments import ComprehensiveExperiments
    runner = ComprehensiveExperiments()
    agents = runner.import_agents()
    runner.experiment_2_hyperparameters(
        agents=agents,
        base_timesteps=50000  
    )
    print("\n Hyperparameter experiments complete!")


def run_convergence_experiments():
    print("\n" + "="*70)
    print("RUNNING CONVERGENCE EXPERIMENTS")
    print("="*70)
    from experiments.run_all_experiments import ComprehensiveExperiments
    runner = ComprehensiveExperiments()
    agents = runner.import_agents()
    runner.experiment_3_convergence(
        agents=agents,
        eval_intervals=8,
        total_steps=100000
    )
    print("\n Convergence experiments complete!")


def analyze_results():
    print("\n" + "="*70)
    print("ANALYZING RESULTS")
    print("="*70)
    os.system('python experiments/analyze_results.py')


def generate_plots():
    print("\n" + "="*70)
    print("GENERATING DOCUMENTATION PLOTS")
    print("="*70)
    os.system('python experiments/generate_documentation_plots.py')


def compare_agents():
    print("\n" + "="*70)
    print("RUNNING HEAD-TO-HEAD COMPARISON")
    print("="*70)
    os.system('python agents_comparison/compare_all.py')


def main():
    parser = argparse.ArgumentParser(description='Quick Experiment Runner')
    parser.add_argument('--baseline-only', action='store_true',
                       help='Run only baseline experiments')
    parser.add_argument('--hyperparams-only', action='store_true',
                       help='Run only hyperparameter experiments')
    parser.add_argument('--convergence-only', action='store_true',
                       help='Run only convergence experiments')
    parser.add_argument('--skip-plots', action='store_true',
                       help='Skip plot generation')
    parser.add_argument('--skip-comparison', action='store_true',
                       help='Skip head-to-head comparison')
    args = parser.parse_args()
    start_time = datetime.now()
    print("\n" + "="*70)
    print("QUICK EXPERIMENT RUNNER (OPTIMIZED)")
    print("="*70)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nConfiguration:")
    print("  • Seeds: 3 (reduced from 5)")
    print("  • Timesteps: 100k (reduced from 500k)")
    print("  • Hyperparams: Expanded grid")
    print("  • Plots: 10 comprehensive visualizations")
    print("="*70 + "\n")
    try:
        if args.baseline_only:
            run_baseline_experiments()
        elif args.hyperparams_only:
            run_hyperparameter_experiments()
        elif args.convergence_only:
            run_convergence_experiments()
        else:
            run_baseline_experiments()
            run_hyperparameter_experiments()
            response = input("\nRun convergence analysis? (adds ~30 min) [y/N]: ")
            if response.lower() == 'y':
                run_convergence_experiments()
        if not args.baseline_only and not args.hyperparams_only:
            analyze_results()
            if not args.skip_plots:
                generate_plots()
            if not args.skip_comparison:
                compare_agents()
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60
        print("\n" + "="*70)
        print(" ALL EXPERIMENTS COMPLETE!")
        print("="*70)
        print(f"\nTotal time: {duration:.1f} minutes")
        print(f"\n Results saved to:")
        print(f"  • experiments/results/")
        print(f"  • experiments/plots/")
        print(f"  • documentation_plots/")
        print(f"\n Generated outputs:")
        print(f"  • Baseline comparison results")
        print(f"  • Hyperparameter sensitivity analysis")
        print(f"  • 10+ publication-quality plots")
        print(f"  • Statistical analysis report")
        print(f"  • Head-to-head comparison")
        print("\n Next steps:")
        print("  1. Review plots in documentation_plots/")
        print("  2. Check report in experiments/results/report.txt")
        print("  3. Use plots for documentation/presentation")
        print("="*70 + "\n")
    except KeyboardInterrupt:
        print("\n\n  Interrupted by user")
        print("Partial results may be available in experiments/results/")
    except Exception as e:
        print(f"\n\n Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
