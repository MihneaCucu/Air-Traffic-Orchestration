# Air Traffic Control - Reinforcement Learning Project

## 👥 Authors

* **Izabela Jilavu**
* **Mihnea Cucu**
* **Antonio Soare**
* **Cezar Tulceanu**
* **Cristina Cârstea**

---

## 1. Introduction

Air traffic control represents a complex sequential optimization problem under uncertainty, requiring the efficient coordination of multiple dynamic entities (airplanes) in a restricted space (runways) with strict temporal constraints (takeoff/landing windows) and stochastic events (unplanned arrivals). This inherent complexity makes the ATC domain an ideal candidate for applying Reinforcement Learning techniques.

### 1.1 Motivation and Context
* **Limitations of traditional methods**: The traditional approach based on predefined rules and manual procedures presents significant limitations in dynamic and unforeseen situations.
* **The power of RL**: Reinforcement Learning offers a promising alternative through its ability to learn optimal policies directly from interaction with the environment, without requiring the explicit specification of control strategies.
* **Problem formulation as MDP**: We model the problem as a Markov Decision Process (MDP) defined by the tuple (S, A, P, R, gamma):
    * **S**: State space (9-dimensional observations).
    * **A**: Action space (3 discrete actions).
    * **P**: Transition function (determined by environment dynamics).
    * **R**: Reward function (designed for multi-objective optimization).
    * **gamma = 0.99**: Discount factor for future rewards.

### 1.2 What did we build?
1.  **The Environment**:
    * A 2D airport with 2 parallel runways.
    * A FIFO queue with a maximum capacity of 12 airplanes in preparation for takeoff.
    * Airplanes that appear randomly (p=0.25) and want to land, blocking the runways.
    * Graphical visualization using `pygame`.
2.  **The Agents (Learning Agents)**:
    * **Random Agent**: Chooses actions randomly (baseline).
    * **DQN (Deep Q-Network)**: Learns action values (value-based).
    * **PPO (Proximal Policy Optimization)**: Learns directly which policy to follow (policy-based).
    * **A2C (Advantage Actor-Critic)**: Synchronous actor-critic implementation.
    * **SAC (Soft Actor-Critic)**: Maximum entropy soft actor-critic.
    * **Rainbow DQN**: High-performance agent that combines six major improvements to the standard DQN.
3.  **Comprehensive Experiments**:
    * Tests with over 50 hyperparameter configurations.
    * Statistical analyses using 3 different seeds for reproducibility.

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
