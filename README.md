# ✈️ Air Traffic Control - Reinforcement Learning Project

A sophisticated multi-agent reinforcement learning environment for simulating air traffic control operations with realistic physics, weather effects, and safety constraints.

## 🎯 Project Overview

This project implements an **Air Traffic Control (ATC) simulation environment** where RL agents learn to manage aircraft takeoffs and landings across multiple runways while avoiding collisions, managing fuel constraints, and handling dynamic weather conditions.

**Team Members:** [Add your names here]

---

## 📋 Problem Formulation

### State Space (9 dimensions)
- `planes_in_queue` - Number of aircraft waiting for departure
- `dep_occupied[0]` - Runway 1 departure status
- `dep_y[0]` - Runway 1 departure position
- `dep_occupied[1]` - Runway 2 departure status
- `dep_y[1]` - Runway 2 departure position
- `arrival_active` - Incoming aircraft status
- `arrival_lane` - Arrival runway assignment
- `arrival_y` - Arrival aircraft position
- `arrivals_landed` - Total successful landings

### Action Space (3 discrete actions)
- **0**: Wait/Hold - No action taken
- **1**: Clear Runway 1 - Authorize departure or landing
- **2**: Clear Runway 2 - Authorize departure or landing

### Reward Structure
| Event | Reward | Rationale |
|-------|--------|-----------|
| Time step | -0.01 | Encourage efficiency |
| Critical fuel | -0.2/plane/step | Prevent fuel emergencies |
| Departure cleared | +1.0 | Positive action feedback |
| Successful takeoff | +15.0 | Main objective |
| Successful landing | +15.0 | Main objective |
| Minor violation | -5.0 | Discourage unsafe operations |
| Critical violation | -10.0 | Strong penalty for danger |
| Episode completion | +100.0 | Bonus for clearing all aircraft |

### Environment Dynamics
- **Weather System**: Wind speed affects aircraft movement (0-60 km/h)
- **Fuel Management**: Each queued aircraft has fuel timer (critical < 15 steps)
- **Stochastic Arrivals**: 20% chance of incoming aircraft when clearing runway
- **Safety Zones**: Proximity violations occur when aircraft are < 2 units apart
- **Physics Simulation**: Realistic takeoff/landing with ground roll and climb phases

---

## 🏗️ Project Structure

```
RL/
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── train_dqn.py                 # Quick training script
├── visualize.py                 # Quick visualization script
├── run_tests.py                 # Environment testing
│
├── src/                         # Source code
│   ├── environment/             # Environment implementation
│   │   └── atc_env.py          # Main ATC environment
│   │
│   ├── agents/                  # RL Agent implementations
│   │   ├── custom_dqn_agent.py # Custom DQN implementation
│   │   └── model.py            # Neural network architectures
│   │
│   ├── training/                # Training scripts
│   │   ├── train_custom_dqn.py # DQN training
│   │   ├── train.py            # Multi-agent training
│   │   └── run_experiments.py  # Hyperparameter experiments
│   │
│   ├── evaluation/              # Evaluation & analysis
│   │   ├── eval.py             # Model evaluation
│   │   ├── visualize_agent.py  # Agent visualization
│   │   ├── compare_agents.py   # Multi-agent comparison
│   │   └── analyze_experiments.py # Results analysis
│   │
│   └── visualization/           # Plotting utilities
│       ├── generate_all_plots.py
│       ├── generate_hyperparameter_plots.py
│       └── generate_documentation_plots.py
│
├── models/                      # Trained models
│   └── custom_dqn_atc.pth
│
├── logs/                        # Training logs
│   ├── tensorboard/            # TensorBoard logs
│   └── atc_logs/               # Custom logs
│
├── results/                     # Experiment results
│   ├── experiments/            # Raw experiment data
│   └── plots/                  # Generated visualizations
│
├── tests/                       # Unit tests
│   └── test_enhanced_ui.py
│
└── docs/                        # Documentation
    └── documentation.tex
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Key Dependencies
- `gymnasium` - RL environment framework
- `pygame` - Visualization and rendering
- `numpy` - Numerical computations
- `torch` - Deep learning framework
- `stable-baselines3` - RL algorithms library
- `matplotlib` - Plotting
- `tensorboard` - Training visualization

---

## 🎮 Usage

### Quick Start - Train DQN Agent
```bash
python train_dqn.py
```

### Visualize Trained Agent
```bash
python visualize.py
```

### Run Environment Tests
```bash
python run_tests.py
```

### Advanced Training

#### Train Custom DQN
```bash
python src/training/train_custom_dqn.py
```

#### Train Multiple Agents (PPO & DQN)
```bash
python src/training/train.py
```

#### Run Hyperparameter Experiments
```bash
python src/training/run_experiments.py
```

### Evaluation & Analysis

#### Evaluate Agent Performance
```bash
python src/evaluation/eval.py
```

#### Compare Multiple Agents
```bash
python src/evaluation/compare_agents.py
```

#### Analyze Experiment Results
```bash
python src/evaluation/analyze_experiments.py
```

#### Generate Plots
```bash
python src/visualization/generate_all_plots.py
```

---

## 🤖 Implemented Agents

### 1. **Custom DQN** (Deep Q-Network)
- **Type**: Value-based, off-policy
- **Architecture**: 3-layer MLP (64-64 hidden units)
- **Features**: Experience replay, target network, ε-greedy exploration
- **Hyperparameters**:
  - Learning rate: 1e-3
  - Gamma: 0.99
  - Batch size: 64
  - Buffer size: 50,000
  - Target update frequency: 1,000 steps

### 2. **[Agent 2 - To be implemented by teammate]**
- Type: [e.g., Policy-based, Actor-Critic]
- Details: [Add implementation details]

### 3. **[Agent 3 - To be implemented by teammate]**
- Type: [e.g., Model-based, Tabular]
- Details: [Add implementation details]

---

## 📊 Experiments & Results

### Experiment Categories

1. **Baseline Performance**
   - Default hyperparameters
   - Multiple seeds for statistical significance

2. **Hyperparameter Tuning**
   - Learning rate: [1e-4, 1e-3, 1e-2]
   - Gamma (discount factor): [0.95, 0.99, 0.999]
   - Batch size: [32, 64, 128]
   - Target update frequency: [500, 1000, 2000]
   - Epsilon decay: [slow, medium, fast]

3. **Agent Comparison**
   - Same environment conditions
   - Equal training timesteps
   - Consistent evaluation protocol

### Performance Metrics
- **Average Reward**: Mean episodic return
- **Success Rate**: % episodes with all aircraft cleared
- **Safety Score**: Violations per episode
- **Convergence Speed**: Steps to reach threshold performance
- **Stability**: Reward variance across seeds

### Sample Results (Custom DQN)
```
Episode 1: 260.4
Episode 2: 297.4
Episode 3: 264.8
Episode 4: 278.4
Episode 5: 259.1

Average: 272.0 ± 15.8
```

---

## 🎨 Enhanced UI Features

### Visual Elements
- ✅ **Dynamic gradient backgrounds** adapting to weather
- ✅ **Animated clouds** with parallax effect
- ✅ **Realistic rain effects** with wind influence
- ✅ **3D altitude rendering** with dynamic shadows
- ✅ **Exhaust trails** for departing aircraft
- ✅ **Modern information panels** with live statistics

### Information Display
- **Left Panel**: Arrivals statistics with aircraft icons
- **Top Center**: Weather, safety score, violations
- **Bottom Panel**: Departure queue with fuel indicators
- **Real-time Feedback**: Action notifications and alerts
- **Proximity Warnings**: Visual alerts for near-misses

### Rendering Improvements
- 8 frames/step smooth animations
- 800x700 window resolution
- Professional color schemes
- Rounded UI elements
- Status-coded information (red/yellow/green)

---

## 📈 Grading Criteria Alignment

| Category | Points | Status |
|----------|--------|--------|
| **Theme & Problem** (1p) | | |
| Clear RL-relevant theme | 0.5p | ✅ |
| Well-defined state/action/reward | 0.5p | ✅ |
| **Environment** (2p) | | |
| Functional & correct | 1p | ✅ |
| Significant modifications/custom | 0.5p | ✅ |
| Good reward design & dynamics | 0.5p | ✅ |
| **Algorithms** (3p) | | |
| 3+ correct implementations | 2p | 🔄 1/3 done |
| Algorithm diversity | 0.5p | 🔄 Pending |
| Fair comparison | 0.5p | 🔄 Pending |
| **Experiments** (2p) | | |
| Multiple seeds/experiments | 1p | ✅ |
| Hyperparameter analysis | 0.5p | ✅ |
| Stability/convergence discussion | 0.5p | 🔄 Pending |
| **Results** (2p) | | |
| Graphs/tables | 1p | 🔄 Pending |
| Interpretation | 1p | 🔄 Pending |
| **Documentation** (2p) | | |
| Structured documentation | 1p | 🔄 In progress |
| Coherent presentation | 1p | 🔄 Pending |
| **Bonus** (+1p) | | |
| Advanced features | +1p | 🔄 Candidate |

---

## 🎯 Next Steps

### For Teammates
1. **Implement additional agents** in `src/agents/`:
   - Suggested: PPO (policy-based), SARSA (tabular), A3C (actor-critic)
   - Follow the structure of `custom_dqn_agent.py`
   - Ensure compatibility with `ATC2DEnv`

2. **Test your agent**:
   ```python
   from src.environment import ATC2DEnv
   from src.agents import YourAgent
   
   env = ATC2DEnv()
   agent = YourAgent(env)
   agent.learn(total_timesteps=500000)
   agent.save("models/your_agent.pth")
   ```

3. **Run comparison experiments**:
   - Use `src/evaluation/compare_agents.py`
   - Document results in `results/experiments/`

### For Final Presentation
1. ✅ Complete all agent implementations (3+ minimum)
2. ✅ Run comprehensive hyperparameter experiments
3. ✅ Generate comparison plots and tables
4. ✅ Write analysis and interpretation
5. ✅ Prepare presentation (6-7 minutes)
6. ✅ Upload code and documentation

---

## 📝 Development Notes

### Recent Improvements
- **Reward rebalancing** for better learning signal
- **Enhanced UI** with professional graphics
- **Clean project structure** for team collaboration
- **Automated testing** framework
- **Comprehensive logging** for analysis

### Known Issues
- Virtual environment (`rl-env/`) may need recreation
- High variance in early training (expected for DQN)
- Weather effects can occasionally cause outliers

### Future Enhancements
- Multi-agent scenarios (simultaneous controllers)
- Additional weather conditions (fog, ice)
- More complex airspace (3+ runways)
- Real-world data integration

---

## 📚 References

- Sutton & Barto - Reinforcement Learning: An Introduction
- Stable-Baselines3 Documentation
- Gymnasium Environment Framework
- DQN Paper: Mnih et al. (2015)

---

## 📞 Contact

**Course**: Reinforcement Learning
**Academic Year**: 2025-2026
**Institution**: [Your University]

For questions or contributions, contact team members:
- [Member 1]: [email]
- [Member 2]: [email]
- [Member 3]: [email]

---

## 📄 License

This project is developed for academic purposes.

---

**Last Updated**: January 14, 2026
