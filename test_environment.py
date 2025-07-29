#!/usr/bin/env python3
"""
Simple test script to verify the UAV environment works correctly.
Run this after installing the required packages.
"""

try:
    from rl import UAVEnvironment
    import numpy as np
    
    def test_environment():
        print("Testing UAV Environment...")
        
        # Create environment
        env = UAVEnvironment()
        print(f"✓ Environment created successfully")
        print(f"  - Action space shape: {env.action_space.shape}")
        print(f"  - Observation space shape: {env.observation_space.shape}")
        print(f"  - Number of users: {env.num_users}")
        print(f"  - Number of subchannels: {env.num_subchannels}")
        
        # Test reset
        obs, info = env.reset()
        print(f"✓ Environment reset successfully")
        print(f"  - Observation shape: {obs.shape}")
        print(f"  - UAV initial position: {env.uav_position}")
        
        # Test random action
        action = env.action_space.sample()
        print(f"✓ Random action sampled: shape {action.shape}")
        
        # Test step
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✓ Environment step successful")
        print(f"  - Reward: {reward}")
        print(f"  - Users served: {reward} out of {env.num_users}")
        print(f"  - UAV position after step: {env.uav_position}")
        
        # Test multiple steps
        total_reward = reward
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
        print(f"✓ Multiple steps completed")
        print(f"  - Total reward over 6 steps: {total_reward}")
        print(f"  - Final UAV position: {env.uav_position}")
        
        # Test power allocations
        if hasattr(env, 'power_allocations'):
            print(f"✓ Power allocations computed")
            print(f"  - Power allocation shape: {env.power_allocations.shape}")
            print(f"  - Total power per user (first 5): {np.sum(env.power_allocations[:5], axis=1)}")
        
        print("\n🎉 All tests passed! The environment is working correctly.")
        
    if __name__ == "__main__":
        test_environment()
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install the required packages:")
    print("pip install -r requirements.txt")
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
