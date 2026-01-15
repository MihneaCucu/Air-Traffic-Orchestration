import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import numpy as np
import torch
torch.set_num_threads(1)

from src.environment.atc_env import ATC2DEnv
from sac_agent import DiscreteSAC, ReplayBuffer
from datetime import datetime
import argparse


def evaluate_agent(agent, env, n_episodes=5, verbose=False):
    episode_rewards = []
    episode_lengths = []
    successful_episodes = 0

    for i in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        terminated = False

        while not done:
            action = agent.select_action(state, deterministic=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = next_state
            episode_reward += reward
            episode_length += 1

        if terminated:
            successful_episodes += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        if verbose:
            status = "" if terminated else ""
            print(f"  {status} Episode {i+1}/{n_episodes}: "
                  f"Reward={episode_reward:.2f}, Length={episode_length}")

    mean_reward = np.mean(episode_rewards)
    mean_length = np.mean(episode_lengths)
    success_rate = (successful_episodes / n_episodes) * 100

    return mean_reward, mean_length, success_rate


def train_sac(
    total_timesteps=500000,
    eval_freq=10000,
    save_freq=50000,
    batch_size=256,
    buffer_size=100000,
    learning_starts=1000,
    lr=3e-4,
    gamma=0.99,
    tau=0.005,
    hidden_dims=None,
    seed=None
):

    if hidden_dims is None:
        hidden_dims = [256, 256]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = script_dir
    log_dir = os.path.join(os.path.dirname(script_dir), "atc_logs")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(f"{models_dir}/sac_checkpoints", exist_ok=True)

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    env = ATC2DEnv()
    eval_env = ATC2DEnv()

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 70)
    print("SAC TRAINING FOR ATC ENVIRONMENT")
    print("=" * 70)
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print(f"Device: {device}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Learning starts: {learning_starts:,}")
    print(f"Batch size: {batch_size}")
    print(f"Buffer size: {buffer_size:,}")
    print("=" * 70)

    agent = DiscreteSAC(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=lr,
        gamma=gamma,
        tau=tau,
        alpha=0.2,
        auto_entropy_tuning=True,
        hidden_dims=hidden_dims,
        device=device
    )

    replay_buffer = ReplayBuffer(capacity=buffer_size)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = f"{log_dir}/sac_training_{timestamp}.log"

    episode_rewards = []
    episode_lengths = []
    timestep = 0
    episode = 0
    best_eval_reward = -np.inf

    print("\n Starting training...")
    print(f" Logs: {log_file}")
    print("-" * 70)

    state, _ = env.reset()
    episode_reward = 0
    episode_length = 0

    total_q1_loss = 0
    total_q2_loss = 0
    total_policy_loss = 0
    update_count = 0

    while timestep < total_timesteps:
        if timestep < learning_starts:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state, deterministic=False)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        replay_buffer.push(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward
        episode_length += 1
        timestep += 1

        if timestep >= learning_starts:
            losses = agent.update(replay_buffer, batch_size=batch_size)

            if losses:
                total_q1_loss += losses['q1_loss']
                total_q2_loss += losses['q2_loss']
                total_policy_loss += losses['policy_loss']
                update_count += 1

        if done:
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode += 1

            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_length = np.mean(episode_lengths[-10:])

                avg_q1 = total_q1_loss / max(update_count, 1)
                avg_q2 = total_q2_loss / max(update_count, 1)
                avg_policy = total_policy_loss / max(update_count, 1)

                print(f"Episode {episode:4d} | Step {timestep:7d} | "
                      f"Reward: {episode_reward:7.2f} | Avg(10): {avg_reward:7.2f} | "
                      f"Len: {episode_length:3d} | "
                      f"Q1Loss: {avg_q1:.4f} | PolLoss: {avg_policy:.4f}")

                total_q1_loss = 0
                total_q2_loss = 0
                total_policy_loss = 0
                update_count = 0

            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0

        if timestep % eval_freq == 0 and timestep > 0:
            print("\n" + "=" * 70)
            print(f" EVALUATION at step {timestep:,}")
            print("-" * 70)

            eval_reward, eval_length, success_rate = evaluate_agent(
                agent, eval_env, n_episodes=5, verbose=True
            )

            print("-" * 70)
            print(f"Mean Reward: {eval_reward:.2f}")
            print(f"Mean Length: {eval_length:.2f}")
            print(f"Success Rate: {success_rate:.1f}%")
            print("=" * 70 + "\n")

            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                best_model_path = f"{models_dir}/sac_atc_best.pth"
                agent.save(best_model_path)
                print(f"🏆 New best model! Saved to {best_model_path}\n")

        if timestep % save_freq == 0 and timestep > 0:
            checkpoint_path = f"{models_dir}/sac_checkpoints/sac_atc_{timestep}.pth"
            agent.save(checkpoint_path)

    final_model_path = f"{models_dir}/sac_atc.pth"
    agent.save(final_model_path)

    print("\n" + "=" * 70)
    print(" TRAINING COMPLETE!")
    print("=" * 70)
    print(f" Final model: {final_model_path}")
    print(f" Best model: {models_dir}/sac_atc_best.pth")
    print(f" Checkpoints: {models_dir}/sac_checkpoints/")
    print("=" * 70)

    print("\n Final Evaluation (10 episodes)...")
    print("-" * 70)
    final_reward, final_length, final_success = evaluate_agent(
        agent, eval_env, n_episodes=10, verbose=True
    )
    print("-" * 70)
    print(f"Final Mean Reward: {final_reward:.2f}")
    print(f"Final Mean Length: {final_length:.2f}")
    print(f"Final Success Rate: {final_success:.1f}%")
    print("=" * 70)

    env.close()
    eval_env.close()

    return agent


def main():
    parser = argparse.ArgumentParser(description='Train SAC agent on ATC environment')
    parser.add_argument('--quick', action='store_true', help='Quick training (100k steps)')
    parser.add_argument('--long', action='store_true', help='Long training (1M steps)')
    parser.add_argument('--timesteps', type=int, default=None, help='Custom timesteps')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')

    args = parser.parse_args()

    if args.timesteps is not None:
        total_timesteps = args.timesteps
    elif args.quick:
        print(" Quick training mode (100k timesteps)\n")
        total_timesteps = 100000
    elif args.long:
        print(" Long training mode (1M timesteps)\n")
        total_timesteps = 1000000
    else:
        print(" Full training mode (500k timesteps)\n")
        total_timesteps = 500000

    train_sac(
        total_timesteps=total_timesteps,
        eval_freq=10000,
        save_freq=50000,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
