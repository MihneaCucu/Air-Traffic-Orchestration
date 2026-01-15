import sys
sys.path.insert(0, '.')

from src.environment import ATC2DEnv
from DQN.custom_dqn_agent import CustomDQN
from PPO.ppo_agent import PPOAgent
from A2C.agent_a2c import A2CAgent
from SAC.sac_agent import DiscreteSAC

AGENT_TYPE = "SAC" 
EPISODES = 3

env = ATC2DEnv(render_mode="human")

print(f"Loading {AGENT_TYPE} agent...")

if AGENT_TYPE == "DQN":
    model_path = "experiments/models/DQN_baseline_seed0.pth"
    agent = CustomDQN(env)
    agent.load(model_path)
elif AGENT_TYPE == "PPO":
    model_path = "experiments/models/PPO_baseline_seed0.pth"
    agent = PPOAgent(env)
    agent.load(model_path)
elif AGENT_TYPE == "A2C":
    model_path = "experiments/models/A2C_baseline_seed0.pth"
    agent = A2CAgent(env)
    agent.load(model_path)
elif AGENT_TYPE == "SAC":
    model_path = "experiments/models/SAC_baseline_seed0.pth"
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DiscreteSAC(state_dim, action_dim)
    agent.load(model_path)
else:
    raise ValueError(f"Unknown agent type: {AGENT_TYPE}")

print(f"Model loaded from: {model_path}")
print(f"Running {EPISODES} episodes with visualization...\n")

for ep in range(EPISODES):
    obs, _ = env.reset()
    done = False
    truncated = False
    score = 0
    steps = 0
    print(f"=== Episode {ep + 1}/{EPISODES} ===")
    while not (done or truncated):
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        score += reward
        steps += 1
    print(f"Final Score: {score:.2f}")
    print(f"Steps: {steps}")
    print(f"Planes departed: {info.get('planes_departed', 'N/A')}")
    print(f"Planes landed: {info.get('planes_landed', 'N/A')}\n")

env.close()
print("Visualization complete!")
