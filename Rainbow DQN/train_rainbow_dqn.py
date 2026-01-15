import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.environment import ATC2DEnv
from src.agents import RainbowDQN

MODELS_DIR = "models"
LOG_DIR = "atc_logs"


def train():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    env = ATC2DEnv()

    agent = RainbowDQN(
        env,
        learning_rate=1e-4,
        gamma=0.99,
        batch_size=64,
        buffer_size=100000,
        target_update_freq=2000,
        train_start=2000,
        train_freq=1,
        gradient_steps=1,
        n_step=3,
        per_alpha=0.6,
        per_beta_start=0.4,
        per_beta_frames=500000,
        atoms=51,
        v_min=-50.0,
        v_max=150.0,
        hidden_dim=128,
        log_dir=LOG_DIR,
    )

    agent.learn(total_timesteps=500000)

    save_path = os.path.join(MODELS_DIR, "rainbow_dqn_atc.pth")
    agent.save(save_path)
    print(f"Rainbow DQN salvat cu succes la {save_path}")


if __name__ == "__main__":
    train()
