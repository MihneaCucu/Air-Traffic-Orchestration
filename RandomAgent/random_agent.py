import numpy as np


class RandomAgent:
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
    def predict(self, observation, deterministic=False):
        action = self.action_space.sample()
        return action, None
    def learn(self, total_timesteps=None):
        print(f"RandomAgent: No training required (random policy)")
        pass
    def save(self, path):
        print(f"RandomAgent: No model to save (random policy)")
        pass
    def load(self, path):
        print(f"RandomAgent: No model to load (random policy)")
        pass
    def get_action_probabilities(self):
        n_actions = self.action_space.n
        return {i: 1.0 / n_actions for i in range(n_actions)}
    def __repr__(self):
        return f"RandomAgent(action_space={self.action_space})"
