import numpy as np
import matplotlib.pyplot as plt
import time
from collections import deque
import os

# Import the improved environment from the main file
import sys
sys.path.append('/Users/alekramelaheehridoy/Desktop/projects/abid')
from train_uav_constrained import UAVEnvironmentConstrained, TrainingCallback

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

def run_short_training_demo():
    """Run a shorter training demo to show the improvements"""
    print("UAV Training Demo - Short Version")
    print("="*60)
    print("Running 50,000 timesteps (50 episodes) for demonstration")
    print("="*60)
    
    # Create environment
    env = UAVEnvironmentConstrained()
    env = Monitor(env)
    
    # Define PPO model with improved parameters
    policy_kwargs = dict(
        net_arch=[512, 512, 256],
        activation_fn=torch.nn.ReLU,
    )
    
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=0.0003,  # Increased learning rate
        gamma=0.99,
        gae_lambda=0.98,
        n_steps=1000,  # Match episode length
        batch_size=1000,
        n_epochs=10,
        clip_range=0.2,
        ent_coef=0.02,  # Increased entropy for exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=None  # Disable tensorboard for demo
    )
    
    # Custom callback for tracking
    callback = TrainingCallback(env, save_freq=5, plot_freq=10)
    
    # Train for shorter period
    print("\nStarting training...")
    start_time = time.time()
    model.learn(total_timesteps=50000, callback=callback)
    training_time = time.time() - start_time
    
    print(f"\nTraining completed in {training_time:.1f} seconds")
    
    # Get the unwrapped environment
    base_env = env.env if hasattr(env, 'env') else env
    
    # Plot results
    plot_demo_results(base_env)
    
    # Test the trained model
    print("\n" + "="*60)
    print("Testing Trained Model")
    print("="*60)
    
    test_results = test_trained_model(model, base_env)
    
    # Save model
    model.save("ppo_uav_demo_model")
    print("\nModel saved to 'ppo_uav_demo_model'")
    
    return base_env, model, test_results

def plot_demo_results(env):
    """Plot training results from the demo"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    episodes = range(1, len(env.episode_rewards) + 1)
    
    # Reward curve
    ax = axes[0, 0]
    ax.plot(episodes, env.episode_rewards, 'b-', alpha=0.7, linewidth=2)
    ax.scatter(episodes, env.episode_rewards, c='blue', s=50, alpha=0.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Reward')
    ax.set_title('Training Reward Curve')
    ax.grid(True, alpha=0.3)
    
    # Add trend line
    if len(episodes) > 5:
        z = np.polyfit(list(episodes), env.episode_rewards, 1)
        p = np.poly1d(z)
        ax.plot(episodes, p(episodes), "r--", alpha=0.8, label=f'Trend: {z[0]:.2f}x + {z[1]:.1f}')
        ax.legend()
    
    # Users served
    ax = axes[0, 1]
    ax.plot(episodes, env.episode_users_served, 'g-', alpha=0.7, linewidth=2)
    ax.scatter(episodes, env.episode_users_served, c='green', s=50, alpha=0.5)
    ax.axhline(y=85, color='r', linestyle='--', label='Target: 85 users')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Users Served')
    ax.set_title('Users Served Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])
    
    # Power efficiency
    ax = axes[0, 2]
    ax.plot(episodes, np.array(env.episode_power_efficiency) * 100, 'm-', alpha=0.7, linewidth=2)
    ax.scatter(episodes, np.array(env.episode_power_efficiency) * 100, c='magenta', s=50, alpha=0.5)
    ax.axhline(y=90, color='r', linestyle='--', label='Target: 90%')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Power Efficiency (%)')
    ax.set_title('Power Efficiency Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])
    
    # Altitude violations
    ax = axes[1, 0]
    ax.plot(episodes, env.episode_altitude_violations, 'r-', alpha=0.7, linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Altitude Violations')
    ax.set_title('Altitude Constraint Violations')
    ax.grid(True, alpha=0.3)
    
    # Velocity violations
    ax = axes[1, 1]
    ax.plot(episodes, env.episode_velocity_violations, 'orange', alpha=0.7, linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Velocity Violations')
    ax.set_title('Velocity Constraint Violations')
    ax.grid(True, alpha=0.3)
    
    # Combined metrics
    ax = axes[1, 2]
    ax2 = ax.twinx()
    
    l1 = ax.plot(episodes, env.episode_users_served, 'g-', label='Users Served', linewidth=2)
    l2 = ax2.plot(episodes, np.array(env.episode_power_efficiency) * 100, 'm-', label='Power Efficiency %', linewidth=2)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Users Served', color='g')
    ax2.set_ylabel('Power Efficiency (%)', color='m')
    ax.tick_params(axis='y', labelcolor='g')
    ax2.tick_params(axis='y', labelcolor='m')
    ax.set_title('Combined Performance Metrics')
    ax.grid(True, alpha=0.3)
    
    # Add legend
    lns = l1 + l2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='center right')
    
    plt.tight_layout()
    plt.savefig('demo_training_results.png', dpi=150)
    plt.close()
    
    print("Training results saved to 'demo_training_results.png'")

def test_trained_model(model, env):
    """Test the trained model on different scenarios"""
    print("\nTesting on different user distributions...")
    
    scenarios = ['uniform', 'clustered', 'edge', 'mixed']
    results = {}
    
    for scenario in scenarios:
        print(f"\n{scenario.capitalize()} distribution:")
        
        # Reset with specific distribution
        obs, _ = env.reset(options={'reset_position': True, 'user_distribution': scenario})
        
        episode_metrics = {
            'rewards': [],
            'users_served': [],
            'power_efficiency': [],
            'trajectory': [env.uav_position.copy()]
        }
        
        # Run one episode
        for step in range(200):  # Shorter test episode
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_metrics['rewards'].append(reward)
            episode_metrics['users_served'].append(info['users_served'])
            episode_metrics['power_efficiency'].append(info['power_efficiency'])
            episode_metrics['trajectory'].append(env.uav_position.copy())
            
            if terminated or truncated:
                break
        
        avg_users = np.mean(episode_metrics['users_served'])
        avg_efficiency = np.mean(episode_metrics['power_efficiency'])
        avg_reward = np.mean(episode_metrics['rewards'])
        
        print(f"  - Average users served: {avg_users:.1f}")
        print(f"  - Average power efficiency: {avg_efficiency:.2%}")
        print(f"  - Average reward: {avg_reward:.1f}")
        
        results[scenario] = {
            'metrics': episode_metrics,
            'user_positions': env.users_positions.copy(),
            'trajectory': np.array(episode_metrics['trajectory'])
        }
    
    # Plot test results
    plot_test_results(results)
    
    return results

def plot_test_results(results):
    """Plot testing results showing UAV behavior on different distributions"""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    for idx, (scenario, data) in enumerate(results.items()):
        # Top view trajectory
        ax = axes[0, idx]
        trajectory = data['trajectory']
        users = data['user_positions']
        
        # Plot users
        ax.scatter(users[:, 0], users[:, 1], c='lightgray', s=30, alpha=0.5, label='Users')
        
        # Plot trajectory with color gradient
        for i in range(len(trajectory) - 1):
            color = plt.cm.viridis(i / len(trajectory))
            ax.plot(trajectory[i:i+2, 0], trajectory[i:i+2, 1], color=color, linewidth=2)
        
        ax.scatter(trajectory[0, 0], trajectory[0, 1], 
                  color='green', s=150, marker='o', edgecolors='black', label='Start', zorder=5)
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], 
                  color='red', s=150, marker='*', edgecolors='black', label='End', zorder=5)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'{scenario.capitalize()} Distribution')
        ax.set_xlim([0, 1000])
        ax.set_ylim([0, 1000])
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend()
        
        # Performance over time
        ax = axes[1, idx]
        timesteps = range(len(data['metrics']['users_served']))
        ax.plot(timesteps, data['metrics']['users_served'], 'g-', label='Users Served', linewidth=2)
        ax.plot(timesteps, np.array(data['metrics']['power_efficiency']) * 100, 
                'm--', label='Power Eff. %', linewidth=2)
        
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Value')
        ax.set_title(f'Performance - {scenario.capitalize()}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig('demo_test_results.png', dpi=150)
    plt.close()
    
    print("\nTest results saved to 'demo_test_results.png'")

if __name__ == "__main__":
    env, model, test_results = run_short_training_demo()
    
    # Final summary
    print("\n" + "="*60)
    print("DEMO SUMMARY")
    print("="*60)
    
    if len(env.episode_rewards) > 0:
        print(f"Training Episodes Completed: {len(env.episode_rewards)}")
        print(f"Final Performance:")
        print(f"  - Last episode reward: {env.episode_rewards[-1]:.1f}")
        print(f"  - Last episode users served: {env.episode_users_served[-1]:.1f}")
        print(f"  - Last episode power efficiency: {env.episode_power_efficiency[-1]:.2%}")
        
        if len(env.episode_rewards) > 10:
            print(f"\nImprovement over training:")
            print(f"  - Reward: {env.episode_rewards[0]:.1f} → {env.episode_rewards[-1]:.1f}")
            print(f"  - Users: {env.episode_users_served[0]:.1f} → {env.episode_users_served[-1]:.1f}")
            print(f"  - Efficiency: {env.episode_power_efficiency[0]:.2%} → {env.episode_power_efficiency[-1]:.2%}")
    
    print("\n✅ Key improvements implemented:")
    print("  ✓ Dynamic UAV movement towards underserved areas")
    print("  ✓ Smart power allocation based on channel quality")
    print("  ✓ Exploration bonuses for discovering new areas")
    print("  ✓ Penalties for wasting power on unservable users")
    print("  ✓ Altitude optimization based on user density")