# Air Traffic Control - Reinforcement Learning Project

## 👥 Authors

* [cite_start]**Izabela Jilavu** [cite: 2]
* [cite_start]**Mihnea Cucu** [cite: 2]
* [cite_start]**Antonio Soare** [cite: 2]
* [cite_start]**Cezar Tulceanu** [cite: 2]
* [cite_start]**Cristina Cârstea** [cite: 2]
* [cite_start]*University of Bucharest* [cite: 2]

---

## 1. Introduction

[cite_start]Air traffic control represents a complex sequential optimization problem under uncertainty, requiring the efficient coordination of multiple dynamic entities (airplanes) in a restricted space (runways) with strict temporal constraints (takeoff/landing windows) and stochastic events (unplanned arrivals)[cite: 12]. [cite_start]This inherent complexity makes the ATC domain an ideal candidate for applying Reinforcement Learning techniques[cite: 13].

### 1.1 Motivation and Context
* [cite_start]**Limitations of traditional methods**: The traditional approach based on predefined rules and manual procedures presents significant limitations in dynamic and unforeseen situations[cite: 14].
* [cite_start]**The power of RL**: Reinforcement Learning offers a promising alternative through its ability to learn optimal policies directly from interaction with the environment, without requiring the explicit specification of control strategies[cite: 15].
* [cite_start]**Problem formulation as MDP**: We model the problem as a Markov Decision Process (MDP) defined by the tuple $(S, A, P, R, \gamma)$[cite: 16]:
    * [cite_start]**$S$**: State space (9-dimensional observations)[cite: 17].
    * [cite_start]**$A$**: Action space (3 discrete actions)[cite: 18].
    * [cite_start]**$P$**: Transition function (determined by environment dynamics)[cite: 19].
    * [cite_start]**$R$**: Reward function (designed for multi-objective optimization)[cite: 20].
    * [cite_start]**$\gamma = 0.99$**: Discount factor for future rewards[cite: 21].

### 1.2 What did we build?
1.  [cite_start]**The Environment**: [cite: 28]
    * [cite_start]A 2D airport with 2 parallel runways[cite: 29].
    * [cite_start]A FIFO queue with a maximum capacity of 12 airplanes in preparation for takeoff[cite: 50].
    * [cite_start]Airplanes that appear randomly ($p=0.25$) and want to land, blocking the runways[cite: 55].
    * [cite_start]Graphical visualization using `pygame`[cite: 32].
2.  [cite_start]**The Agents (Learning Agents)**: [cite: 33]
    * [cite_start]**Random Agent**: Chooses actions randomly (baseline)[cite: 34].
    * [cite_start]**DQN (Deep Q-Network)**: Learns action values (value-based)[cite: 35].
    * [cite_start]**PPO (Proximal Policy Optimization)**: Learns directly which policy to follow (policy-based)[cite: 36].
    * [cite_start]**A2C (Advantage Actor-Critic)**: Synchronous actor-critic implementation[cite: 37, 192].
    * [cite_start]**SAC (Soft Actor-Critic)**: Maximum entropy soft actor-critic[cite: 39, 281].
    * [cite_start]**Rainbow DQN**: High-performance agent that combines six major improvements to the standard DQN[cite: 424].
3.  [cite_start]**Comprehensive Experiments**: [cite: 40]
    * [cite_start]Tests with over 50 hyperparameter configurations[cite: 42].
    * [cite_start]Statistical analyses using 3 different seeds for reproducibility[cite: 43].

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
