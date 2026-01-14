"""
Test rapid pentru SAC agent
Verifică că toate componentele funcționează
"""

import os
# Fix PyTorch threading issues on macOS
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import numpy as np
import torch
torch.set_num_threads(1)

from atc_env import ATC2DEnv
from sac_agent import DiscreteSAC, ReplayBuffer

def test_components():
    """Test individual components"""
    print("=" * 60)
    print("TEST SAC COMPONENTS")
    print("=" * 60)

    # Test environment
    print("\n1. Testing ATC Environment...")
    env = ATC2DEnv()
    state, _ = env.reset()
    print(f"   ✓ Observation shape: {state.shape}")
    print(f"   ✓ Action space: {env.action_space}")

    # Test replay buffer
    print("\n2. Testing Replay Buffer...")
    buffer = ReplayBuffer(capacity=1000)
    for i in range(100):
        buffer.push(state, 0, 1.0, state, False)
    print(f"   ✓ Buffer size: {len(buffer)}")

    states, actions, rewards, next_states, dones = buffer.sample(32)
    print(f"   ✓ Sample batch shapes: {states.shape}, {actions.shape}")

    # Test SAC agent
    print("\n3. Testing SAC Agent...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = DiscreteSAC(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        device=device
    )
    print(f"   ✓ Agent created on device: {device}")

    # Test action selection
    action = agent.select_action(state, deterministic=False)
    print(f"   ✓ Action selected: {action}")

    # Test update
    for _ in range(50):
        buffer.push(state, action, 1.0, state, False)

    losses = agent.update(buffer, batch_size=32)
    print(f"   ✓ Update successful")
    print(f"     Q1 Loss: {losses['q1_loss']:.4f}")
    print(f"     Policy Loss: {losses['policy_loss']:.4f}")
    print(f"     Alpha: {losses['alpha']:.4f}")

    # Test save/load
    print("\n4. Testing Save/Load...")
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "test_model.pth")
        agent.save(model_path)
        print(f"   ✓ Model saved")

        agent2 = DiscreteSAC(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            device=device
        )
        agent2.load(model_path)
        print(f"   ✓ Model loaded")

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nSAC agent is ready to train!")
    print("\nNext steps:")
    print("  python train_sac.py --quick    # Quick training")
    print("  python eval_sac.py             # Evaluate model")
    print("=" * 60)

def test_mini_training():
    """Run a very short training to verify everything works"""
    print("\n" + "=" * 60)
    print("MINI TRAINING TEST (100 steps)")
    print("=" * 60)

    env = ATC2DEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = DiscreteSAC(state_dim=state_dim, action_dim=action_dim, device=device)
    buffer = ReplayBuffer(capacity=1000)

    state, _ = env.reset()
    total_reward = 0

    for step in range(100):
        # Random exploration first
        if step < 20:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state, deterministic=False)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        buffer.push(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward

        # Update agent
        if len(buffer) > 32:
            losses = agent.update(buffer, batch_size=32)

        if done:
            state, _ = env.reset()

        if (step + 1) % 20 == 0:
            print(f"  Step {step+1}/100: Buffer size={len(buffer)}, Total reward={total_reward:.2f}")

    print("\n✓ Mini training completed successfully!")
    print(f"  Final buffer size: {len(buffer)}")
    print(f"  Total reward: {total_reward:.2f}")
    print("=" * 60)

if __name__ == "__main__":
    # Run tests
    test_components()

    # Optional: mini training test
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--mini-train":
        test_mini_training()

