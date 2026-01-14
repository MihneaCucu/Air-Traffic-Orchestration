#!/usr/bin/env python
"""
Script de Test Complet pentru SAC
==================================

Verifică că toate componentele SAC funcționează corect.
"""

import os
import sys
# Add parent directory to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

print("=" * 70)
print("TEST COMPLET SAC")
print("=" * 70)

# Test 1: Import-uri
print("\n[1/6] Testing imports...")
try:
    import torch
    print(f"  ✓ PyTorch {torch.__version__}")
    from sac_agent import DiscreteSAC, ReplayBuffer
    print("  ✓ SAC agent imported")
    from src.environment.atc_env import ATC2DEnv
    print("  ✓ ATC environment imported")
except Exception as e:
    print(f"  ✗ Import failed: {e}")
    exit(1)

# Test 2: Environment
print("\n[2/6] Testing environment...")
try:
    env = ATC2DEnv()
    state, _ = env.reset()
    print(f"  ✓ Environment created")
    print(f"  ✓ State shape: {state.shape}")
    print(f"  ✓ Action space: {env.action_space.n} actions")
    env.close()
except Exception as e:
    print(f"  ✗ Environment failed: {e}")
    exit(1)

# Test 3: Replay Buffer
print("\n[3/6] Testing replay buffer...")
try:
    buffer = ReplayBuffer(capacity=1000)
    for i in range(100):
        buffer.push(state, 0, 1.0, state, False)
    print(f"  ✓ Replay buffer works")
    print(f"  ✓ Buffer size: {len(buffer)}")
    states, actions, rewards, next_states, dones = buffer.sample(32)
    print(f"  ✓ Sampling works")
except Exception as e:
    print(f"  ✗ Replay buffer failed: {e}")
    exit(1)

# Test 4: Create SAC Agent (critical test!)
print("\n[4/6] Testing SAC agent creation...")
print("  → This is where it usually blocks...")
try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  → Using device: {device}")

    agent = DiscreteSAC(
        state_dim=9,
        action_dim=3,
        device=device
    )
    print("  ✓ SAC agent created successfully!")
except Exception as e:
    print(f"  ✗ SAC agent creation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 5: Action Selection
print("\n[5/6] Testing action selection...")
try:
    env = ATC2DEnv()
    state, _ = env.reset()

    # Test deterministic action
    action = agent.select_action(state, deterministic=True)
    print(f"  ✓ Deterministic action: {action}")

    # Test stochastic action
    action = agent.select_action(state, deterministic=False)
    print(f"  ✓ Stochastic action: {action}")

    env.close()
except Exception as e:
    print(f"  ✗ Action selection failed: {e}")
    exit(1)

# Test 6: Model Save/Load
print("\n[6/6] Testing model save/load...")
try:
    import os
    import tempfile

    # Save
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        temp_path = f.name

    agent.save(temp_path)
    print(f"  ✓ Model saved")

    # Load
    agent2 = DiscreteSAC(
        state_dim=9,
        action_dim=3,
        device='cpu'
    )
    agent2.load(temp_path)
    print(f"  ✓ Model loaded")

    # Cleanup
    os.remove(temp_path)
    print(f"  ✓ Cleanup done")

except Exception as e:
    print(f"  ✗ Save/load failed: {e}")
    exit(1)

# Success!
print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED!")
print("=" * 70)
print("\nSAC is working correctly. You can now:")
print("  • Evaluate models:    python eval_sac.py --best")
print("  • Visualize agent:    python visualize_sac.py --best")
print("  • Train new model:    python train_sac.py --quick")
print("=" * 70)

