"""
Script de Evaluare pentru SAC Agent pe ATC Environment
======================================================

Evaluează agentul SAC antrenat pe environment-ul ATC.

Usage:
------
python eval_sac.py                      # Evaluează modelul final
python eval_sac.py --best               # Evaluează best model
python eval_sac.py --render             # Cu vizualizare
python eval_sac.py --episodes 50        # Număr custom de episoade
python eval_sac.py --model path.pth     # Model custom
"""

import os
import sys
# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Fix PyTorch threading issues on macOS
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import numpy as np
import torch
torch.set_num_threads(1)

import argparse
from atc_env import ATC2DEnv
from sac_agent import DiscreteSAC


def evaluate_sac(model_path, n_episodes=100, render=False, verbose=True):
    """
    Evaluează un agent SAC antrenat

    Args:
        model_path: Path către modelul salvat (.pth)
        n_episodes: Număr de episoade de evaluare
        render: Dacă True, afișează environment-ul
        verbose: Dacă True, printează detalii

    Returns:
        dict cu statistici de evaluare
    """

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    render_mode = "human" if render else None
    env = ATC2DEnv(render_mode=render_mode)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Load agent
    if verbose:
        print("=" * 70)
        print("EVALUARE SAC AGENT")
        print("=" * 70)
        print(f"Model: {model_path}")
        print(f"Device: {device}")
        print(f"Environment: {env.observation_space}, {env.action_space}")
        print(f"Episoade: {n_episodes}")
        print("=" * 70)

    if not os.path.exists(model_path):
        print(f"❌ Model nu a fost găsit: {model_path}")
        return None

    agent = DiscreteSAC(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device
    )

    try:
        agent.load(model_path)
    except Exception as e:
        print(f"❌ Eroare la încărcarea modelului: {e}")
        return None

    # Evaluate
    episode_rewards = []
    episode_lengths = []
    successful_episodes = 0

    if verbose:
        print("\n🎯 Running evaluation...")
        print("-" * 70)

    for ep in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            # Select action (deterministic pentru evaluare)
            action = agent.select_action(state, deterministic=True)

            # Step environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = next_state
            episode_reward += reward
            episode_length += 1

        # Track success
        if terminated:  # Success = terminated natural, nu truncated
            successful_episodes += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # Print progress
        if verbose and (ep + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            print(f"Episode {ep+1:3d}/{n_episodes}: "
                  f"Avg Reward (last 10): {avg_reward:7.2f}, "
                  f"Avg Length: {avg_length:5.1f}")

    # Calculate statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    min_reward = np.min(episode_rewards)
    max_reward = np.max(episode_rewards)

    mean_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)

    success_rate = (successful_episodes / n_episodes) * 100

    # Print summary
    if verbose:
        print("-" * 70)
        print("\n" + "=" * 70)
        print("📊 REZULTATE EVALUARE")
        print("=" * 70)
        print(f"Episoade evaluate: {n_episodes}")
        print(f"\nReward Statistics:")
        print(f"  Mean Reward:    {mean_reward:7.2f} ± {std_reward:.2f}")
        print(f"  Min Reward:     {min_reward:7.2f}")
        print(f"  Max Reward:     {max_reward:7.2f}")
        print(f"\nEpisode Statistics:")
        print(f"  Mean Length:    {mean_length:7.2f} ± {std_length:.2f}")
        print(f"  Success Rate:   {success_rate:6.1f}% ({successful_episodes}/{n_episodes})")
        print("=" * 70)

        # Interpretation
        print("\n💡 Interpretare:")
        if mean_reward > 100:
            print("  ✅ Agent excelent! Reward pozitiv mare.")
        elif mean_reward > 0:
            print("  ✓ Agent decent. Reward pozitiv.")
        elif mean_reward > -50:
            print("  ⚠ Agent mediu. Reward aproape de 0.")
        else:
            print("  ❌ Agent slab. Reward negativ.")

        if success_rate > 80:
            print("  ✅ Success rate foarte bun!")
        elif success_rate > 50:
            print("  ✓ Success rate decent.")
        else:
            print("  ⚠ Success rate scăzut.")

        print("=" * 70)

    env.close()

    return {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'min_reward': min_reward,
        'max_reward': max_reward,
        'mean_length': mean_length,
        'std_length': std_length,
        'success_rate': success_rate,
        'successful_episodes': successful_episodes,
        'total_episodes': n_episodes,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths
    }


def compare_models(n_episodes=50):
    """Compară toate modelele SAC disponibile"""

    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = script_dir

    models_to_compare = {
        'SAC Final': f"{models_dir}/sac_atc.pth",
        'SAC Best': f"{models_dir}/sac_atc_best.pth",
    }

    # Check for checkpoints
    checkpoint_dir = f"{models_dir}/sac_checkpoints"
    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
        if checkpoints:
            # Add latest checkpoint
            checkpoints.sort()
            latest = checkpoints[-1]
            model_paths[f'SAC Checkpoint ({latest})'] = f"{checkpoint_dir}/{latest}"

    print("=" * 70)
    print("🔍 COMPARARE MODELE SAC")
    print("=" * 70)

    results = {}

    for name, path in model_paths.items():
        if os.path.exists(path):
            print(f"\n📁 Evaluez: {name}")
            print("-" * 70)
            result = evaluate_sac(path, n_episodes=n_episodes, render=False, verbose=False)

            if result:
                results[name] = result
                print(f"  Mean Reward: {result['mean_reward']:7.2f} ± {result['std_reward']:.2f}")
                print(f"  Success Rate: {result['success_rate']:5.1f}%")
        else:
            print(f"\n⚠ {name} nu există: {path}")

    # Summary
    if results:
        print("\n" + "=" * 70)
        print("📊 REZUMAT COMPARATIV")
        print("=" * 70)

        # Sort by mean reward
        sorted_results = sorted(results.items(), key=lambda x: x[1]['mean_reward'], reverse=True)

        for i, (name, stats) in enumerate(sorted_results, 1):
            print(f"\n{i}. {name}")
            print(f"   Reward:  {stats['mean_reward']:7.2f} ± {stats['std_reward']:.2f}")
            print(f"   Success: {stats['success_rate']:6.1f}%")
            print(f"   Length:  {stats['mean_length']:7.2f}")

        print("\n" + "=" * 70)
        best = sorted_results[0]
        print(f"🏆 BEST MODEL: {best[0]}")
        print(f"   Mean Reward: {best[1]['mean_reward']:.2f}")
        print(f"   Success Rate: {best[1]['success_rate']:.1f}%")
        print("=" * 70)

    return results


def main():
    """Main function cu argument parsing"""

    parser = argparse.ArgumentParser(
        description='Evaluate SAC agent on ATC environment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python eval_sac.py                    # Evaluate final model
  python eval_sac.py --best             # Evaluate best model
  python eval_sac.py --render           # With visualization
  python eval_sac.py --compare          # Compare all models
  python eval_sac.py --episodes 50      # Custom episodes
  python eval_sac.py --model path.pth   # Custom model
        """
    )

    parser.add_argument('--model', type=str, default=None,
                       help='Custom model path')
    parser.add_argument('--best', action='store_true',
                       help='Evaluate best model instead of final')
    parser.add_argument('--compare', action='store_true',
                       help='Compare all available models')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of evaluation episodes (default: 100)')
    parser.add_argument('--render', action='store_true',
                       help='Render environment during evaluation')

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = script_dir

    if args.compare:
        # Compare all models
        compare_models(n_episodes=args.episodes)
    else:
        # Evaluate single model
        if args.model:
            model_path = args.model
        elif args.best:
            model_path = f"{models_dir}/sac_atc_best.pth"
        else:
            model_path = f"{models_dir}/sac_atc.pth"
            # Fallback to best if final doesn't exist
            if not os.path.exists(model_path):
                print(f"⚠ Final model nu există, încerc best model...")
                model_path = f"{models_dir}/sac_atc_best.pth"

        # Evaluate
        result = evaluate_sac(
            model_path=model_path,
            n_episodes=args.episodes,
            render=args.render,
            verbose=True
        )

        if result is None:
            print("\n❌ Evaluarea a eșuat!")
            print("\nModele disponibile:")

            # List available models
            if os.path.exists(f"{models_dir}/sac_atc.pth"):
                print(f"  ✓ {models_dir}/sac_atc.pth")
            if os.path.exists(f"{models_dir}/sac_atc_best.pth"):
                print(f"  ✓ {models_dir}/sac_atc_best.pth")

            checkpoint_dir = f"{models_dir}/sac_checkpoints"
            if os.path.exists(checkpoint_dir):
                checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
                if checkpoints:
                    print(f"  ✓ {len(checkpoints)} checkpoint(s) în {checkpoint_dir}/")

            print("\nPentru antrenare, rulează:")
            print("  python train_sac.py --quick    # Quick (100k steps)")
            print("  python train_sac.py            # Full (500k steps)")
            print("  python train_sac.py --long     # Long (1M steps)")


if __name__ == "__main__":
    main()

