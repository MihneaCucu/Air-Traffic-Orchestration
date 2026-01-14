"""
Test script for the enhanced ATC environment UI
This will run the environment without an agent to showcase the visual improvements
"""

import gymnasium as gym
from atc_env import ATC2DEnv
import numpy as np

def main():
    print("=" * 60)
    print("✈  ENHANCED ATC CONTROL TOWER - UI TEST")
    print("=" * 60)
    print("\nFeatures:")
    print("  • Beautiful gradient backgrounds")
    print("  • Dynamic weather effects (clouds, rain, wind)")
    print("  • Enhanced plane graphics with shadows")
    print("  • Modern information panels")
    print("  • Smooth animations (8 frames per step)")
    print("  • Real-time status feedback")
    print("  • Safety score tracking")
    print("  • Proximity alerts with visual warnings")
    print("\nClose the window to exit.\n")
    
    # Create environment with rendering
    env = ATC2DEnv(
        queue_length=12,
        lane_height=10,
        max_steps=500,
        arrival_prob=0.02,
        render_mode="human"
    )
    
    episodes = 3
    
    for ep in range(episodes):
        print(f"\n--- Episode {ep + 1}/{episodes} ---")
        obs, _ = env.reset()
        done = False
        truncated = False
        score = 0
        steps = 0
        
        while not (done or truncated):
            # Simple policy: alternate between runways and wait
            if steps % 15 < 5:
                action = 0  # Wait
            elif steps % 15 < 10:
                action = 1  # Use runway 1
            else:
                action = 2  # Use runway 2
            
            # Random action occasionally for variety
            if np.random.rand() < 0.1:
                action = np.random.randint(0, 3)
            
            obs, reward, done, truncated, info = env.step(action)
            score += reward
            steps += 1
            
            # Print status every 50 steps
            if steps % 50 == 0:
                print(f"  Step {steps}: Score = {score:.1f}, "
                      f"Queue = {int(obs[0])}, "
                      f"Landed = {int(obs[8])}")
        
        print(f"Episode {ep + 1} finished!")
        print(f"  Final Score: {score:.2f}")
        print(f"  Total Steps: {steps}")
        print(f"  Planes Landed: {env.arrivals_landed}")
        print(f"  Total Departures: {env.total_departures}")
        print(f"  Safety Score: {env.safety_score}/100")
        print(f"  Near Misses: {env.near_misses}")
    
    env.close()
    print("\n" + "=" * 60)
    print("Test completed! The environment is ready to use.")
    print("=" * 60)

if __name__ == "__main__":
    main()
