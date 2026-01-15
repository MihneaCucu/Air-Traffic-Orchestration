# Air Traffic Control - Reinforcement Learning Project

Reinforcement learning agents for air traffic control simulation.

## Setup

### Create Virtual Environment

```bash
python3 -m venv rl-env
source rl-env/bin/activate
```

On Windows:
```bash
python -m venv rl-env
rl-env\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Train an Agent

```bash
cd DQN
python train_dqn.py
```

Or:
```bash
cd A2C
python train_a2c.py
```

Or:
```bash
cd PPO
python train_ppo.py
```

Or:
```bash
cd SAC
python train_sac.py
```

### Visualize Trained Agent

```bash
python visualize.py
```

### Run Experiments

```bash
python experiments/run_all_experiments.py
```

### Analyze Results

```bash
python experiments/analyze_results.py
```

## Project Structure

```
RL/
├── README.md
├── requirements.txt
├── visualize.py
├── quick_experiments.py
├── A2C/
├── DQN/
├── PPO/
├── SAC/
├── RandomAgent/
├── Rainbow DQN/
├── experiments/
│   ├── run_all_experiments.py
│   ├── analyze_results.py
│   ├── results/
│   ├── models/
│   └── logs/
└── src/
    └── environment/
```

## Requirements

- Python 3.8+
- gymnasium
- pygame
- numpy
- torch
- matplotlib
- tensorboard
