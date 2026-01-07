# Air Traffic Control - Reinforcement Learning Project

## Structure

### Core Files
- `atc_env.py` - Environment implementation (Gymnasium)
- `custom_dqn_agent.py` - Manual DQN implementation
- `model.py`, `train.py` - Stable Baselines3 agents (PPO, DQN)
- `train_custom_dqn.py` - Train Custom DQN
- `eval.py`, `compare_agents.py` - Evaluation and comparison
- `visualize_agent.py` - Visual behavior analysis
- `documentation.tex` - LaTeX documentation

### Experiment Scripts
- `run_experiments.py` - 27 hyperparameter experiments (2-4h)
- `run_quick_experiments.py` - 6 quick experiments (30-60min)
- `analyze_experiments.py` - Generate analysis and plots
- `generate_plots_auto.py` - Generate all documentation plots

### Directories
- `models/` - Saved models (.pth, .zip)
- `models/experiments/` - Experiment models (27 files)
- `atc_logs/` - TensorBoard logs
- `experiment_results/` - Experiment results (JSON, reports, plots)
- `documentation_plots/` - Documentation plots (PNG)
- `rl-env/` - Python virtual environment

## Quick Start

### Setup
```bash
source rl-env/bin/activate
```

### Training
```bash
# Custom DQN
python train_custom_dqn.py

# PPO (Stable Baselines3)
python train.py
```

### Experiments
```bash
# Run 27 hyperparameter experiments
python run_experiments.py

# Analyze results
python analyze_experiments.py

# Monitor training
tensorboard --logdir atc_logs
```

### Generate Plots
```bash
python generate_plots_auto.py
```

Output in `documentation_plots/`:
- `learning_curves.png` - Training curves
- `final_performance.png` - Final performance comparison
- `hyperparameter_impact.png` - Hyperparameter analysis
- `training_stability.png` - Stability across seeds
- `convergence_speed.png` - Convergence comparison
- `comprehensive_summary.png` - Complete overview

### Evaluation
```bash
# Compare agents
python compare_agents.py

# Detailed evaluation
python eval.py
```

## Experiments Configuration

Tested hyperparameters (9 configurations x 3 seeds = 27 runs):
- Learning Rate: 1e-4, 1e-3, 5e-3
- Gamma: 0.95, 0.99, 0.995
- Batch Size: 32, 64, 128
- Target Update Frequency: 500, 1000
- Epsilon Decay: 0.995, 0.998

Best configuration identified: freq_target_update (reward 168.57 ± 0.14)

## Results

### Experiment Results
```
experiment_results/
├── comparison.png - Comparative plots
├── report.txt - Detailed text report
├── all_experiments_summary.json - All results summary
└── [config]_seed[N]_results.json - Individual results (27 files)
```

### Models
```
models/
├── custom_dqn_atc.pth - Final trained Custom DQN
├── ppo_atc.zip - Stable Baselines3 PPO
└── experiments/ - Experiment models (27 files)
```

### TensorBoard Logs
```
atc_logs/
├── CustomDQN_[timestamp]/ - Custom DQN runs
├── PPO_LongRun_[N]/ - PPO runs
└── DQN_run_[N]/ - Stable Baselines3 DQN runs
```

## Cleanup

```bash
# Run interactive cleanup script
./cleanup.sh

# Or manual cleanup
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
rm -f *.aux *.log *.out missfont.log
```

Keep: `experiment_results/`, `documentation_plots/`, `models/custom_dqn_atc.pth`, `models/*.zip`
Delete: `__pycache__/`, `*.pyc`, LaTeX temp files, optionally `models/experiments/`

## Team Workflow

### Adding New Agent
1. Create agent file (e.g., `my_agent.py`)
2. Add training script (e.g., `train_my_agent.py`)
3. Run experiments with different seeds
4. Update `documentation.tex` in your subsection
5. Generate plots: `python generate_plots_auto.py`
6. Update this guide with your agent commands

### Running Your Experiments
```python
# Template
from my_agent import MyAgent
from atc_env import ATC2DEnv

env = ATC2DEnv()
agent = MyAgent()
agent.train(env, total_timesteps=300000, seed=42)
agent.save("models/my_agent.pth")

# Evaluate
rewards = agent.evaluate(env, episodes=20)
print(f"Mean reward: {np.mean(rewards):.2f}")
```

### Regenerate All Plots
```bash
python generate_plots_auto.py
```

Include plots:
```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.9\textwidth]{documentation_plots/learning_curves.png}
\caption{Learning curves comparison}
\label{fig:learning_curves}
\end{figure}
```

## Notes

- Adjust `total_timesteps` in training scripts for longer training (recommended 500k-1M)
- Each experiment run takes ~5-8 minutes (depends on hardware)
- TensorBoard available at http://localhost:6006
- All plots generated at 300 DPI for publication quality
- Experiment results include mean ± std across 3 seeds
