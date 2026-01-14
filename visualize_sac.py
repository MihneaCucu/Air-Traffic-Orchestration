"""
Vizualizare Agent ATC - Suportă SAC, PPO, DQN
=============================================

Script pentru vizualizarea performanței agenților antrenați în timp real.

Usage:
------
python visualize_sac.py                 # Default: SAC
python visualize_sac.py --agent sac     # SAC agent
python visualize_sac.py --agent ppo     # PPO agent
python visualize_sac.py --agent dqn     # DQN agent
python visualize_sac.py --episodes 10   # Custom număr episoade
python visualize_sac.py --best          # Folosește best model (pentru SAC)
python visualize_sac.py --speed slow    # Slow motion pentru debug
"""

# CRITICAL: Set environment variables BEFORE any other imports!
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Import and configure torch BEFORE stable_baselines3
import torch
torch.set_num_threads(1)

import argparse
from atc_env import ATC2DEnv
# NOTE: stable_baselines3 imported lazily in functions to avoid blocking
from sac_agent import DiscreteSAC


def load_sac_agent(model_path):
    """Încarcă agent SAC"""
    env = ATC2DEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    agent = DiscreteSAC(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device
    )

    agent.load(model_path)
    env.close()
    return agent


def visualize_agent(agent, agent_type, n_episodes=5, render_speed='normal'):
    """
    Vizualizează agentul jucând în environment

    Args:
        agent: Agent antrenat (SAC, PPO sau DQN)
        agent_type: Tip agent ('sac', 'ppo', 'dqn')
        n_episodes: Număr de episoade de vizualizat
        render_speed: 'slow', 'normal', 'fast'
    """

    # Setare FPS bazat pe speed
    fps_map = {
        'slow': 5,
        'normal': 10,
        'fast': 20
    }
    fps = fps_map.get(render_speed, 10)

    # Create environment cu rendering
    env = ATC2DEnv(render_mode="human")
    env.metadata["render_fps"] = fps

    print("=" * 70)
    print(f"VIZUALIZARE AGENT {agent_type.upper()}")
    print("=" * 70)
    print(f"Episoade: {n_episodes}")
    print(f"Render speed: {render_speed} ({fps} FPS)")
    print("=" * 70)
    print("\n⏸️  Închide fereastra pentru a opri vizualizarea\n")

    episode_stats = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        score = 0
        steps = 0

        print(f"\n{'='*70}")
        print(f"📺 EPISOD {ep + 1}/{n_episodes}")
        print(f"{'='*70}")

        try:
            while not (done or truncated):
                # Select action bazat pe tip agent
                if agent_type == 'sac':
                    action = agent.select_action(obs, deterministic=True)
                elif agent_type in ['ppo', 'dqn']:
                    action, _ = agent.predict(obs, deterministic=True)
                else:
                    raise ValueError(f"Unknown agent type: {agent_type}")

                # Step environment
                obs, reward, done, truncated, info = env.step(action)
                score += reward
                steps += 1

                # Print periodic updates
                if steps % 50 == 0:
                    print(f"  Step {steps}: Reward={reward:.2f}, Total Score={score:.2f}")

            # Episode finished
            status = "✓ SUCCESS" if done else "⚠ TRUNCATED"
            print(f"\n{status}")
            print(f"  Final Score: {score:.2f}")
            print(f"  Steps: {steps}")
            print(f"{'='*70}")

            episode_stats.append({
                'episode': ep + 1,
                'score': score,
                'steps': steps,
                'success': done
            })

        except KeyboardInterrupt:
            print("\n\n⏹️  Vizualizare oprită de utilizator")
            break
        except Exception as e:
            print(f"\n❌ Eroare în episodul {ep + 1}: {e}")
            break

    env.close()

    # Print summary
    if episode_stats:
        print("\n" + "=" * 70)
        print("📊 REZUMAT VIZUALIZARE")
        print("=" * 70)

        total_episodes = len(episode_stats)
        avg_score = sum(s['score'] for s in episode_stats) / total_episodes
        avg_steps = sum(s['steps'] for s in episode_stats) / total_episodes
        success_count = sum(1 for s in episode_stats if s['success'])
        success_rate = (success_count / total_episodes) * 100

        print(f"Episoade completate: {total_episodes}")
        print(f"Average Score:       {avg_score:.2f}")
        print(f"Average Steps:       {avg_steps:.1f}")
        print(f"Success Rate:        {success_rate:.1f}% ({success_count}/{total_episodes})")
        print("=" * 70)


def main():
    """Main function cu argument parsing"""

    parser = argparse.ArgumentParser(
        description='Visualize ATC agent performance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_sac.py                    # Default: SAC
  python visualize_sac.py --agent ppo        # Visualize PPO
  python visualize_sac.py --agent sac --best # SAC best model
  python visualize_sac.py --episodes 10      # 10 episodes
  python visualize_sac.py --speed slow       # Slow motion
        """
    )

    parser.add_argument('--agent', type=str, choices=['sac', 'ppo', 'dqn'],
                       default='sac', help='Agent type to visualize (default: sac)')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of episodes to visualize (default: 5)')
    parser.add_argument('--best', action='store_true',
                       help='Use best model instead of final (SAC only)')
    parser.add_argument('--model', type=str, default=None,
                       help='Custom model path')
    parser.add_argument('--speed', type=str, choices=['slow', 'normal', 'fast'],
                       default='normal', help='Render speed (default: normal)')

    args = parser.parse_args()

    models_dir = "models"

    # Load agent
    print("=" * 70)
    print("ÎNCĂRCARE MODEL")
    print("=" * 70)

    agent = None
    agent_type = args.agent

    try:
        if args.model:
            # Custom model path
            model_path = args.model
            print(f"Model path: {model_path}")

            if agent_type == 'sac':
                agent = load_sac_agent(model_path)
            elif agent_type == 'ppo':
                from stable_baselines3 import PPO
                agent = PPO.load(model_path)
            elif agent_type == 'dqn':
                from stable_baselines3 import DQN
                agent = DQN.load(model_path)
        else:
            # Default model paths
            if agent_type == 'sac':
                if args.best:
                    model_path = f"{models_dir}/sac_atc_best.pth"
                else:
                    model_path = f"{models_dir}/sac_atc.pth"
                    # Fallback to best if final doesn't exist
                    if not os.path.exists(model_path):
                        model_path = f"{models_dir}/sac_atc_best.pth"

                print(f"Loading SAC from: {model_path}")
                agent = load_sac_agent(model_path)

            elif agent_type == 'ppo':
                from stable_baselines3 import PPO
                model_path = f"{models_dir}/ppo_atc.zip"
                print(f"Loading PPO from: {model_path}")
                agent = PPO.load(model_path)

            elif agent_type == 'dqn':
                from stable_baselines3 import DQN
                model_path = f"{models_dir}/dqn_atc.zip"
                print(f"Loading DQN from: {model_path}")
                agent = DQN.load(model_path)

        print(f"✓ Model încărcat cu succes!")
        print("=" * 70)

        # Visualize
        visualize_agent(agent, agent_type, args.episodes, args.speed)

    except FileNotFoundError as e:
        print(f"\n❌ Model nu a fost găsit!")
        print(f"   Path: {model_path if 'model_path' in locals() else 'N/A'}")
        print("\n📁 Modele disponibile:")

        # List available models
        if agent_type == 'sac':
            if os.path.exists(f"{models_dir}/sac_atc.pth"):
                print(f"  ✓ {models_dir}/sac_atc.pth")
            if os.path.exists(f"{models_dir}/sac_atc_best.pth"):
                print(f"  ✓ {models_dir}/sac_atc_best.pth")
        elif agent_type == 'ppo':
            if os.path.exists(f"{models_dir}/ppo_atc.zip"):
                print(f"  ✓ {models_dir}/ppo_atc.zip")
        elif agent_type == 'dqn':
            if os.path.exists(f"{models_dir}/dqn_atc.zip"):
                print(f"  ✓ {models_dir}/dqn_atc.zip")

        print("\n💡 Pentru antrenare:")
        if agent_type == 'sac':
            print("  python train_sac.py --quick")
        elif agent_type == 'ppo':
            print("  python train.py")

    except Exception as e:
        print(f"\n❌ Eroare la încărcarea modelului: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

