import sys
sys.path.insert(0, '../..')

from stable_baselines3.common.evaluation import evaluate_policy
from src.environment import ATC2DEnv
from stable_baselines3 import PPO

# Facem un mediu de evaluare (fără grafică e mai rapid)
eval_env = ATC2DEnv()
model = PPO.load("ppo_atc")

# Evaluăm
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100)

print(f"Mean Reward: {mean_reward} +/- {std_reward}")
# Dacă Mean Reward este pozitiv (ex: +90), agentul ajunge des la destinație.
# Dacă e negativ (ex: -100), face multe accidente.