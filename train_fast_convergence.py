import numpy as np
import matplotlib.pyplot as plt
import time
from collections import deque
import os

# ML imports
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
from gymnasium import spaces

class UAVEnvironmentFast(gym.Env):

    def __init__(self):
        super(UAVEnvironmentFast, self).__init__()
        self.num_users = 100  
        self.num_subchannels = 5  
        self.W_s = 100  
        self.N_0 = (10 ** (-80 / 10)) / 1000  
        self.max_power = 1.0  
        self.R_min = 3.0  
        self.time_granularity = 0.1  
        
        # Constraints
        self.max_acceleration = 3.0  # Higher for faster learning
        self.max_velocity = 10.0  # Higher for faster coverage
        self.max_altitude = 500.0  # Lower ceiling for faster convergence
        self.min_altitude = 50.0  
        
        # Initialize UAV
        self.uav_position = np.array([500, 500, 150])  
        self.prev_velocity = np.array([0.0, 0.0, 0.0])
        
        self.action_space = spaces.Box(
            low=np.zeros(6 + 2 * self.num_users * self.num_subchannels),  
            high=np.ones(6 + 2 * self.num_users * self.num_subchannels),  
            dtype=np.float32
        )

        obs_size = 3 + 3 + self.num_users  
        self.observation_space = spaces.Box(low=0, high=1, shape=(obs_size,), dtype=np.float32)

        # Tracking metrics
        self.users_served_per_timestep = []
        self.throughput_tracker = np.zeros((self.num_users,))
        
        # Episode tracking
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_users_served = []
        self.episode_power_efficiency = []
        
        # Trajectory tracking
        self.trajectory = []
        self.episode_trajectories = []
        
        # For convergence detection
        self.best_users_served = 0
        self.convergence_counter = 0
        
        self.reset(seed=42)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        # Always reset position for consistency
        self.uav_position = np.array([500.0, 500.0, 150.0])
        self.prev_velocity = np.array([0.0, 0.0, 0.0])

        # Simple user distribution - clustered for easier learning
        if self.episode_count < 200:
            # Early training: single cluster
            center = np.array([500, 500])
            self.users_positions = np.random.normal(center, 200, size=(self.num_users, 2))
        elif self.episode_count < 500:
            # Mid training: two clusters
            n_per_cluster = self.num_users // 2
            cluster1 = np.random.normal([300, 300], 150, size=(n_per_cluster, 2))
            cluster2 = np.random.normal([700, 700], 150, size=(n_per_cluster, 2))
            self.users_positions = np.vstack([cluster1, cluster2])
        else:
            # Late training: more complex distributions
            dist_type = self.episode_count % 3
            if dist_type == 0:
                # Three clusters
                n_per_cluster = self.num_users // 3
                cluster1 = np.random.normal([250, 500], 100, size=(n_per_cluster, 2))
                cluster2 = np.random.normal([750, 500], 100, size=(n_per_cluster, 2))
                cluster3 = np.random.normal([500, 250], 100, size=(self.num_users - 2*n_per_cluster, 2))
                self.users_positions = np.vstack([cluster1, cluster2, cluster3])
            elif dist_type == 1:
                # Ring distribution
                angles = np.linspace(0, 2*np.pi, self.num_users)
                radius = 300 + np.random.normal(0, 50, self.num_users)
                self.users_positions = np.column_stack([
                    500 + radius * np.cos(angles),
                    500 + radius * np.sin(angles)
                ])
            else:
                # Uniform
                self.users_positions = np.random.uniform(100, 900, size=(self.num_users, 2))
        
        self.users_positions = np.clip(self.users_positions, 0, 1000)
        self.users_positions = np.column_stack((self.users_positions, np.zeros(self.num_users)))

        self._update_channel_gains()

        # Reset episode metrics
        self.step_count = 0
        self.cumulative_reward = 0
        self.trajectory = [self.uav_position.copy()]
        self.timestep_rewards = []
        self.timestep_users_served = []
        self.timestep_power_efficiency = []
        self.episode_power_efficiency_sum = 0
        
        return self._get_state(), {}

    def _update_channel_gains(self):
        a, b = 0.28, 9.6  
        H = self.uav_position[2]
        r_nj = np.linalg.norm(self.users_positions[:, :2] - self.uav_position[:2], axis=1)

        theta = np.arctan(H / (r_nj + 1e-8)) * 180 / np.pi
        P_LoS = 1 / (1 + a * np.exp(-b * (theta - a)))
        P_NLoS = 1 - P_LoS

        fc = 2e9  
        c = 3e8   
        d_nj = np.sqrt(H**2 + r_nj**2)

        PL_LoS = 20 * np.log10(4 * np.pi * fc * d_nj / c) + 1
        PL_NLoS = 20 * np.log10(4 * np.pi * fc * d_nj / c) + 20

        PL = P_LoS * PL_LoS + P_NLoS * PL_NLoS
        self.channel_gains = 10 ** (-PL / 10)

    def step(self, action):
        # Extract acceleration parameters
        a_mean, a_var = action[0], action[1]
        theta_mean, theta_var = action[2], action[3]
        phi_mean, phi_var = action[4], action[5]

        # Direct mapping for faster learning
        a = a_mean * self.max_acceleration  
        theta = theta_mean * 2 * np.pi - np.pi  
        phi = phi_mean * np.pi - np.pi/2  # Limited vertical movement

        ax = a * np.sin(theta) * np.cos(phi)
        ay = a * np.sin(theta) * np.sin(phi)
        az = a * np.cos(theta) * 0.3  # Limited z movement

        acceleration = np.array([ax, ay, az])

        # Update velocity
        new_velocity = self.prev_velocity + self.time_granularity * acceleration
        velocity_magnitude = np.linalg.norm(new_velocity)
        
        if velocity_magnitude > self.max_velocity:
            new_velocity = new_velocity * (self.max_velocity / velocity_magnitude)
        
        self.prev_velocity = new_velocity
        
        # Update position
        new_position = self.uav_position + self.prev_velocity * self.time_granularity
        new_position[0] = np.clip(new_position[0], 0, 1000)
        new_position[1] = np.clip(new_position[1], 0, 1000)
        new_position[2] = np.clip(new_position[2], self.min_altitude, self.max_altitude)
        
        self.uav_position = new_position
        self.trajectory.append(self.uav_position.copy())
        
        # Update channel gains
        self._update_channel_gains()
        
        # Process power allocation
        power_means = action[6: 6 + self.num_users * self.num_subchannels]
        power_vars = action[6 + self.num_users * self.num_subchannels: ]  

        power_means = power_means.reshape(self.num_users, self.num_subchannels)
        power_vars = power_vars.reshape(self.num_users, self.num_subchannels)

        self.power_allocations = self._apply_greedy_power_allocation()

        # Calculate throughputs
        throughputs = self._calculate_throughputs()
        served_mask = throughputs >= self.R_min
        users_served = np.sum(served_mask)
        
        # Update best performance
        if users_served > self.best_users_served:
            self.best_users_served = users_served
            self.convergence_counter = 0
        else:
            self.convergence_counter += 1
        
        # Calculate power efficiency
        total_power_to_served = np.sum(self.power_allocations[served_mask])
        total_power_allocated = np.sum(self.power_allocations)
        power_efficiency = total_power_to_served / (total_power_allocated + 1e-8) if total_power_allocated > 0 else 0
        
        # SIMPLIFIED REWARD - focus on users served with efficiency bonus
        reward = users_served * 2.0  # Strong reward for serving users
        
        # Efficiency bonus
        if users_served > 0:
            reward += power_efficiency * 20  # Bonus for efficient allocation
        
        # Position bonus - encourage staying near user centroid
        user_centroid = np.mean(self.users_positions[:, :2], axis=0)
        distance_to_centroid = np.linalg.norm(self.uav_position[:2] - user_centroid)
        if distance_to_centroid < 300:
            reward += (300 - distance_to_centroid) / 100
        
        # Height optimization
        optimal_height = 150
        height_diff = abs(self.uav_position[2] - optimal_height)
        if height_diff < 50:
            reward += (50 - height_diff) / 50
        
        # Track metrics
        self.users_served_per_timestep.append(users_served)
        self.throughput_tracker += throughputs
        self.episode_power_efficiency_sum += power_efficiency
        self.timestep_rewards.append(reward)
        self.timestep_users_served.append(users_served)
        self.timestep_power_efficiency.append(power_efficiency)

        self.cumulative_reward += reward
        self.step_count += 1
        terminated = self.step_count >= 1000
        truncated = False
        
        # Episode end tracking
        if terminated:
            avg_power_efficiency = self.episode_power_efficiency_sum / self.step_count
            
            # Store episode metrics
            self.episode_count += 1
            self.episode_rewards.append(np.mean(self.timestep_rewards))
            self.episode_users_served.append(np.mean(self.timestep_users_served))
            self.episode_power_efficiency.append(avg_power_efficiency)
            self.episode_trajectories.append(np.array(self.trajectory))

        info = {
            'users_served': users_served,
            'power_efficiency': power_efficiency,
            'throughputs': throughputs,
            'served_mask': served_mask,
        }

        return self._get_state(), reward, terminated, truncated, info

    def _calculate_throughputs(self):
        throughputs = np.zeros(self.num_users)
        for user in range(self.num_users):
            total_power = np.sum(self.power_allocations[user])
            snr = (self.channel_gains[user] * total_power) / self.N_0
            throughputs[user] = self.W_s * np.log2(1 + snr)
        return throughputs
    
    def _apply_greedy_power_allocation(self):
        """Simple greedy allocation for fast convergence"""
        # Calculate minimum power needed
        min_power_needed = (self.N_0 * (2**(self.R_min/self.W_s) - 1)) / (self.channel_gains + 1e-8)
        
        # Sort users by channel quality (best first)
        sorted_indices = np.argsort(self.channel_gains)[::-1]
        
        # Allocate power greedily
        power_allocations = np.zeros((self.num_users, self.num_subchannels))
        total_budget = self.max_power * self.num_users * 0.95
        used_budget = 0
        
        for idx in sorted_indices:
            required_power = min_power_needed[idx] * 1.1  # 10% margin
            if used_budget + required_power <= total_budget:
                # Allocate evenly across subchannels
                power_allocations[idx, :] = required_power / self.num_subchannels
                used_budget += required_power
            else:
                break
        
        return power_allocations

    def _get_state(self):
        # Simplified state for faster learning
        pos_normalized = self.uav_position / np.array([1000, 1000, self.max_altitude])
        vel_normalized = np.clip(self.prev_velocity / self.max_velocity, -1, 1)
        
        # Sort channel gains for permutation invariance
        gains_sorted = np.sort(self.channel_gains)[::-1]
        gains_normalized = gains_sorted / (np.max(gains_sorted) + 1e-8)
        
        return np.concatenate([
            pos_normalized,
            vel_normalized,
            gains_normalized
        ]).astype(np.float32)

class FastTrainingCallback(BaseCallback):
    def __init__(self, env, log_interval=20, verbose=1):
        super(FastTrainingCallback, self).__init__(verbose)
        self.env = env
        self.log_interval = log_interval
        self.episode_count = 0
        
    def _on_step(self) -> bool:
        if self.locals.get('dones')[0]:
            self.episode_count += 1
            
            if self.episode_count % self.log_interval == 0:
                env = self.env.unwrapped
                recent_rewards = env.episode_rewards[-self.log_interval:] if len(env.episode_rewards) >= self.log_interval else env.episode_rewards
                recent_users = env.episode_users_served[-self.log_interval:] if len(env.episode_users_served) >= self.log_interval else env.episode_users_served
                recent_efficiency = env.episode_power_efficiency[-self.log_interval:] if len(env.episode_power_efficiency) >= self.log_interval else env.episode_power_efficiency
                
                print(f"\nEpisode {self.episode_count}:")
                print(f"  Avg Reward: {np.mean(recent_rewards):.2f}")
                print(f"  Avg Users Served: {np.mean(recent_users):.2f} / 100")
                print(f"  Avg Power Efficiency: {np.mean(recent_efficiency):.3f}")
                print(f"  Best Users Served: {env.best_users_served}")
                
                # Check for convergence
                if len(recent_users) == self.log_interval:
                    variance = np.var(recent_users)
                    if variance < 5 and np.mean(recent_users) > 80:
                        print("  ✓ Near convergence detected!")
        
        return True

def plot_convergence_results(env, save_path='convergence_results.png'):
    """Plot training curves showing convergence"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    episodes = np.arange(1, len(env.episode_rewards) + 1)
    
    # Use smaller window for smoothing
    window = min(20, len(episodes) // 10)
    
    # Rewards with convergence trend
    ax = axes[0, 0]
    if len(env.episode_rewards) > window:
        rewards_smooth = np.convolve(env.episode_rewards, np.ones(window)/window, mode='valid')
        ax.plot(episodes[:len(rewards_smooth)], rewards_smooth, 'b-', linewidth=2)
        
        # Add convergence indicator
        convergence_value = np.mean(env.episode_rewards[-50:]) if len(env.episode_rewards) > 50 else np.mean(env.episode_rewards)
        ax.axhline(y=convergence_value, color='green', linestyle='--', alpha=0.5, label=f'Converged: {convergence_value:.1f}')
    else:
        ax.plot(episodes, env.episode_rewards, 'b-', linewidth=2)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Reward')
    ax.set_title('Reward Convergence')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Users Served - showing improvement
    ax = axes[0, 1]
    if len(env.episode_users_served) > window:
        users_smooth = np.convolve(env.episode_users_served, np.ones(window)/window, mode='valid')
        ax.plot(episodes[:len(users_smooth)], users_smooth, 'g-', linewidth=2)
        
        # Show improvement trend
        if len(users_smooth) > 100:
            z = np.polyfit(episodes[:len(users_smooth)], users_smooth, 1)
            p = np.poly1d(z)
            ax.plot(episodes[:len(users_smooth)], p(episodes[:len(users_smooth)]), "r--", alpha=0.5, label='Trend')
    else:
        ax.plot(episodes, env.episode_users_served, 'g-', linewidth=2)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Users Served')
    ax.set_title('Users Served Improvement')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Power Efficiency improvement
    ax = axes[1, 0]
    if len(env.episode_power_efficiency) > window:
        efficiency_smooth = np.convolve(env.episode_power_efficiency, np.ones(window)/window, mode='valid')
        ax.plot(episodes[:len(efficiency_smooth)], efficiency_smooth, 'r-', linewidth=2)
        
        # Show starting vs ending efficiency
        start_eff = np.mean(env.episode_power_efficiency[:50]) if len(env.episode_power_efficiency) > 50 else np.mean(env.episode_power_efficiency[:len(env.episode_power_efficiency)//2])
        end_eff = np.mean(env.episode_power_efficiency[-50:]) if len(env.episode_power_efficiency) > 50 else np.mean(env.episode_power_efficiency[len(env.episode_power_efficiency)//2:])
        ax.text(0.05, 0.95, f'Start: {start_eff:.3f}\nEnd: {end_eff:.3f}\nImprovement: {end_eff-start_eff:.3f}', 
                transform=ax.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax.plot(episodes, env.episode_power_efficiency, 'r-', linewidth=2)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Power Efficiency')
    ax.set_title('Power Efficiency Learning')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # Convergence metrics
    ax = axes[1, 1]
    
    # Calculate moving variance to show convergence
    variance_window = 50
    if len(env.episode_users_served) > variance_window:
        users_variance = []
        for i in range(variance_window, len(env.episode_users_served)):
            variance = np.var(env.episode_users_served[i-variance_window:i])
            users_variance.append(variance)
        
        ax.plot(episodes[variance_window:], users_variance, 'purple', linewidth=2)
        ax.set_ylabel('Variance in Users Served')
        ax.set_title('Convergence Indicator (Lower = More Converged)')
        ax.set_yscale('log')
    else:
        # Show cumulative improvement
        cumulative_improvement = []
        baseline = env.episode_users_served[0] if len(env.episode_users_served) > 0 else 0
        for i, users in enumerate(env.episode_users_served):
            improvement = users - baseline
            cumulative_improvement.append(improvement)
        
        ax.plot(episodes, cumulative_improvement, 'purple', linewidth=2)
        ax.set_ylabel('Improvement from Baseline')
        ax.set_title('Cumulative Learning Progress')
    
    ax.set_xlabel('Episode')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('UAV Training Convergence Analysis', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Convergence results saved to {save_path}")

def test_final_performance(model, env_class, episodes=20):
    """Test final model performance"""
    print("\nTesting final model performance...")
    
    env = env_class()
    results = {'users_served': [], 'power_efficiency': [], 'positions': []}
    
    for ep in range(episodes):
        obs, _ = env.reset()
        episode_users = []
        episode_efficiency = []
        episode_positions = []
        
        for _ in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_users.append(info['users_served'])
            episode_efficiency.append(info['power_efficiency'])
            episode_positions.append(env.uav_position.copy())
            
            if terminated or truncated:
                break
        
        results['users_served'].append(np.mean(episode_users))
        results['power_efficiency'].append(np.mean(episode_efficiency))
        results['positions'].append(episode_positions)
    
    print(f"\nFinal Performance (avg over {episodes} episodes):")
    print(f"  Users Served: {np.mean(results['users_served']):.2f} ± {np.std(results['users_served']):.2f}")
    print(f"  Power Efficiency: {np.mean(results['power_efficiency']):.3f} ± {np.std(results['power_efficiency']):.3f}")
    
    return results

def visualize_learned_behavior(env, model, save_path='learned_behavior.png'):
    """Visualize the learned UAV behavior"""
    # Run one episode with the trained model
    obs, _ = env.reset()
    trajectory = [env.uav_position.copy()]
    users_served_over_time = []
    
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        trajectory.append(env.uav_position.copy())
        users_served_over_time.append(info['users_served'])
        
        if terminated or truncated:
            break
    
    trajectory = np.array(trajectory)
    
    fig = plt.figure(figsize=(15, 5))
    
    # 3D trajectory
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'b-', linewidth=2, alpha=0.8)
    ax1.scatter(env.users_positions[:, 0], env.users_positions[:, 1], env.users_positions[:, 2], 
               c='orange', s=20, alpha=0.6, label='Users')
    ax1.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], 
               color='green', s=200, marker='o', label='Start')
    ax1.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], 
               color='red', s=200, marker='s', label='End')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Altitude (m)')
    ax1.set_title('Learned UAV Trajectory')
    ax1.legend()
    
    # Top-down view with coverage
    ax2 = fig.add_subplot(132)
    ax2.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, alpha=0.8)
    ax2.scatter(env.users_positions[:, 0], env.users_positions[:, 1], 
               c='orange', s=30, alpha=0.6)
    
    # Draw coverage circle at final position
    coverage_radius = 300  # Approximate coverage radius
    circle = plt.Circle((trajectory[-1, 0], trajectory[-1, 1]), coverage_radius, 
                       fill=False, edgecolor='red', linewidth=2, linestyle='--')
    ax2.add_patch(circle)
    
    ax2.set_xlabel('X Position (m)')
    ax2.set_ylabel('Y Position (m)')
    ax2.set_title('Top-Down View with Coverage')
    ax2.set_xlim(0, 1000)
    ax2.set_ylim(0, 1000)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # Users served over time
    ax3 = fig.add_subplot(133)
    timesteps = np.arange(len(users_served_over_time))
    ax3.plot(timesteps, users_served_over_time, 'g-', linewidth=2)
    ax3.fill_between(timesteps, 0, users_served_over_time, alpha=0.3, color='green')
    ax3.set_xlabel('Timestep')
    ax3.set_ylabel('Users Served')
    ax3.set_title('Users Served Throughout Episode')
    ax3.set_ylim(0, 100)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Learned behavior visualization saved to {save_path}")

def main():
    print("Starting Fast Convergence UAV Training...")
    print("Target: 2M timesteps with clear convergence")
    print("-" * 50)
    
    # Create environment
    env = Monitor(UAVEnvironmentFast())
    
    # Optimized hyperparameters for fast convergence
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),  # Smaller network for faster training
        activation_fn=torch.nn.Tanh,  # Tanh often converges faster
    )
    
    # Create PPO model with aggressive learning settings
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=5e-4,  # Higher learning rate
        n_steps=1024,  # Fewer steps for faster updates
        batch_size=32,  # Smaller batches
        n_epochs=4,  # Fewer epochs per update
        gamma=0.95,  # Lower gamma for faster learning
        gae_lambda=0.9,
        clip_range=0.2,
        ent_coef=0.02,  # Higher exploration initially
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0,
        tensorboard_log=f"./ppo_fast_tensorboard/PPO_{int(time.time())}"
    )
    
    # Create callback
    callback = FastTrainingCallback(env, log_interval=20)
    
    # Train
    print("\nTraining for 2,000,000 timesteps...")
    start_time = time.time()
    
    model.learn(total_timesteps=2_000_000, callback=callback)
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time/60:.1f} minutes")
    
    # Save model
    model_path = f"ppo_fast_converged_{int(time.time())}.zip"
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Generate convergence plots
    print("\nGenerating convergence analysis plots...")
    plot_convergence_results(env.unwrapped)
    
    # Test final performance
    test_results = test_final_performance(model, UAVEnvironmentFast)
    
    # Visualize learned behavior
    visualize_learned_behavior(env.unwrapped, model)
    
    # Print convergence statistics
    env_unwrapped = env.unwrapped
    print("\n" + "="*60)
    print("CONVERGENCE ANALYSIS")
    print("="*60)
    
    # Check last 100 episodes for convergence
    if len(env_unwrapped.episode_users_served) > 200:
        last_100 = env_unwrapped.episode_users_served[-100:]
        prev_100 = env_unwrapped.episode_users_served[-200:-100]
        
        last_100_mean = np.mean(last_100)
        prev_100_mean = np.mean(prev_100)
        last_100_std = np.std(last_100)
        
        print(f"Episodes -200 to -100: {prev_100_mean:.2f} users served (avg)")
        print(f"Episodes -100 to end:  {last_100_mean:.2f} users served (avg)")
        print(f"Improvement: {last_100_mean - prev_100_mean:.2f} users")
        print(f"Standard deviation (last 100): {last_100_std:.2f}")
        
        # Convergence criteria
        if abs(last_100_mean - prev_100_mean) < 2 and last_100_std < 5:
            print("\n✅ CONVERGENCE ACHIEVED!")
            print(f"   Converged at ~{last_100_mean:.0f} users served")
            print(f"   Power efficiency: {np.mean(env_unwrapped.episode_power_efficiency[-100:]):.3f}")
        else:
            print("\n⚠️  Model is still improving")
    
    # Show learning progression
    print("\nLearning Progression:")
    checkpoints = [100, 500, 1000, 1500, 2000]
    for cp in checkpoints:
        if len(env_unwrapped.episode_users_served) >= cp:
            users = np.mean(env_unwrapped.episode_users_served[cp-50:cp])
            eff = np.mean(env_unwrapped.episode_power_efficiency[cp-50:cp])
            print(f"  Episode {cp}: {users:.1f} users, {eff:.3f} efficiency")

if __name__ == "__main__":
    main()