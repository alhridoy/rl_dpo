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

class UAVEnvironmentDemo(gym.Env):

    def __init__(self):
        super(UAVEnvironmentDemo, self).__init__()
        self.num_users = 100  
        self.num_subchannels = 5  
        self.W_s = 100  
        self.N_0 = (10 ** (-80 / 10)) / 1000  
        self.max_power = 1.0  
        self.R_min = 3.0  
        self.time_granularity = 0.1  
        
        # Constraints optimized for learning
        self.max_acceleration = 5.0  
        self.max_velocity = 15.0  
        self.max_altitude = 400.0  
        self.min_altitude = 80.0  
        
        # Initialize UAV at optimal height
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
        
        self.reset(seed=42)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        # Progressive complexity in user distributions
        if self.episode_count < 50:
            # Very simple: single cluster at center
            center = np.array([500, 500]) + np.random.uniform(-50, 50, 2)
            self.users_positions = np.random.normal(center, 100, size=(self.num_users, 2))
        elif self.episode_count < 150:
            # Two clusters
            n1 = self.num_users // 2
            n2 = self.num_users - n1
            c1 = np.array([400, 400]) + np.random.uniform(-50, 50, 2)
            c2 = np.array([600, 600]) + np.random.uniform(-50, 50, 2)
            cluster1 = np.random.normal(c1, 80, size=(n1, 2))
            cluster2 = np.random.normal(c2, 80, size=(n2, 2))
            self.users_positions = np.vstack([cluster1, cluster2])
        elif self.episode_count < 300:
            # Three clusters
            n_per = self.num_users // 3
            centers = [[300, 300], [700, 300], [500, 700]]
            clusters = []
            for i, center in enumerate(centers):
                n = n_per if i < 2 else self.num_users - 2*n_per
                cluster = np.random.normal(center, 120, size=(n, 2))
                clusters.append(cluster)
            self.users_positions = np.vstack(clusters)
        else:
            # More complex patterns
            pattern = self.episode_count % 4
            if pattern == 0:
                # Ring pattern
                angles = np.linspace(0, 2*np.pi, self.num_users)
                radius = 200 + np.random.normal(0, 30, self.num_users)
                self.users_positions = np.column_stack([
                    500 + radius * np.cos(angles),
                    500 + radius * np.sin(angles)
                ])
            elif pattern == 1:
                # Line pattern
                line_start = np.array([200, 300])
                line_end = np.array([800, 700])
                t_values = np.linspace(0, 1, self.num_users)
                self.users_positions = np.array([
                    line_start + t * (line_end - line_start) + np.random.normal(0, 50, 2)
                    for t in t_values
                ])
            elif pattern == 2:
                # Grid pattern
                grid_size = int(np.sqrt(self.num_users))
                x = np.linspace(200, 800, grid_size)
                y = np.linspace(200, 800, grid_size)
                positions = []
                for i in range(grid_size):
                    for j in range(grid_size):
                        if len(positions) < self.num_users:
                            pos = [x[i], y[j]] + np.random.normal(0, 30, 2)
                            positions.append(pos)
                self.users_positions = np.array(positions)
            else:
                # Uniform random
                self.users_positions = np.random.uniform(100, 900, size=(self.num_users, 2))
        
        # Clip to bounds and add z-coordinate
        self.users_positions = np.clip(self.users_positions, 50, 950)
        self.users_positions = np.column_stack((self.users_positions, np.zeros(self.num_users)))

        # Smart UAV initialization based on users
        user_center = np.mean(self.users_positions[:, :2], axis=0)
        self.uav_position = np.array([user_center[0], user_center[1], 150.0])
        self.prev_velocity = np.array([0.0, 0.0, 0.0])

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
        # Simplified action processing for speed
        a_mean, a_var = action[0], action[1]
        theta_mean, theta_var = action[2], action[3]
        phi_mean, phi_var = action[4], action[5]

        # Direct action mapping (no sampling for speed)
        a = a_mean * self.max_acceleration  
        theta = theta_mean * 2 * np.pi - np.pi  
        phi = phi_mean * np.pi/2 - np.pi/4  # Limited vertical movement

        ax = a * np.sin(theta) * np.cos(phi)
        ay = a * np.sin(theta) * np.sin(phi)
        az = a * np.cos(theta) * 0.2  # Minimal z movement

        acceleration = np.array([ax, ay, az])

        # Update dynamics
        new_velocity = self.prev_velocity + self.time_granularity * acceleration
        velocity_magnitude = np.linalg.norm(new_velocity)
        
        if velocity_magnitude > self.max_velocity:
            new_velocity = new_velocity * (self.max_velocity / velocity_magnitude)
        
        self.prev_velocity = new_velocity
        
        # Update position
        new_position = self.uav_position + self.prev_velocity * self.time_granularity
        new_position[0] = np.clip(new_position[0], 50, 950)
        new_position[1] = np.clip(new_position[1], 50, 950)
        new_position[2] = np.clip(new_position[2], self.min_altitude, self.max_altitude)
        
        self.uav_position = new_position
        self.trajectory.append(self.uav_position.copy())
        
        # Update channel gains
        self._update_channel_gains()
        
        # Smart power allocation
        self.power_allocations = self._apply_smart_power_allocation()

        # Calculate performance
        throughputs = self._calculate_throughputs()
        served_mask = throughputs >= self.R_min
        users_served = np.sum(served_mask)
        
        # Power efficiency
        total_power_to_served = np.sum(self.power_allocations[served_mask])
        total_power_allocated = np.sum(self.power_allocations)
        power_efficiency = total_power_to_served / (total_power_allocated + 1e-8) if total_power_allocated > 0 else 0
        
        # REWARD FUNCTION OPTIMIZED FOR LEARNING
        # Primary: users served (with increasing returns)
        if users_served <= 30:
            reward = users_served * 1.0  # Linear for low performance
        elif users_served <= 70:
            reward = 30 + (users_served - 30) * 2.0  # Higher returns for medium performance
        else:
            reward = 110 + (users_served - 70) * 3.0  # Highest returns for good performance
        
        # Efficiency bonus (scaled with users served)
        if users_served > 10:
            reward += power_efficiency * users_served * 0.2
        
        # Position optimization bonus
        user_centroid = np.mean(self.users_positions[:, :2], axis=0)
        distance_to_centroid = np.linalg.norm(self.uav_position[:2] - user_centroid)
        coverage_bonus = max(0, (400 - distance_to_centroid) / 400) * 10
        reward += coverage_bonus
        
        # Height optimization
        optimal_height = 150
        height_penalty = abs(self.uav_position[2] - optimal_height) * 0.05
        reward -= height_penalty
        
        # Exploration bonus early on
        if self.episode_count < 100:
            movement = np.linalg.norm(self.prev_velocity[:2])
            if movement > 1.0:
                reward += min(movement * 0.5, 5.0)
        
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
    
    def _apply_smart_power_allocation(self):
        """Optimized power allocation for maximum users served"""
        # Calculate minimum power needed for each user
        min_power_needed = (self.N_0 * (2**(self.R_min/self.W_s) - 1)) / (self.channel_gains + 1e-8)
        
        # Sort users by channel quality (descending)
        user_indices = np.arange(self.num_users)
        sorted_indices = user_indices[np.argsort(self.channel_gains)[::-1]]
        
        # Allocate power greedily to best users first
        power_allocations = np.zeros((self.num_users, self.num_subchannels))
        total_budget = self.max_power * self.num_users * 0.95  # Reserve 5% for safety
        used_budget = 0
        
        for user_idx in sorted_indices:
            required_power = min_power_needed[user_idx] * 1.15  # 15% safety margin
            
            if used_budget + required_power <= total_budget:
                # Distribute power evenly across subchannels
                power_per_subchannel = required_power / self.num_subchannels
                power_allocations[user_idx, :] = power_per_subchannel
                used_budget += required_power
            else:
                # Try to allocate remaining budget
                remaining = total_budget - used_budget
                if remaining > min_power_needed[user_idx]:
                    power_allocations[user_idx, :] = remaining / self.num_subchannels
                    used_budget = total_budget
                break
        
        return power_allocations

    def _get_state(self):
        # Efficient state representation
        pos_norm = self.uav_position / np.array([1000, 1000, self.max_altitude])
        vel_norm = np.clip(self.prev_velocity / self.max_velocity, -1, 1)
        
        # Top-k channel gains for efficiency
        top_k = 50  # Only consider top 50 users
        top_gains_idx = np.argsort(self.channel_gains)[-top_k:]
        top_gains = self.channel_gains[top_gains_idx]
        gains_norm = top_gains / (np.max(top_gains) + 1e-8)
        
        # Pad remaining gains with zeros
        gains_padded = np.zeros(self.num_users)
        gains_padded[-top_k:] = gains_norm
        
        return np.concatenate([pos_norm, vel_norm, gains_padded]).astype(np.float32)

class DemoCallback(BaseCallback):
    def __init__(self, env, log_interval=10, verbose=1):
        super(DemoCallback, self).__init__(verbose)
        self.env = env
        self.log_interval = log_interval
        self.episode_count = 0
        
    def _on_step(self) -> bool:
        if self.locals.get('dones')[0]:
            self.episode_count += 1
            
            if self.episode_count % self.log_interval == 0:
                env = self.env.unwrapped
                if len(env.episode_rewards) >= self.log_interval:
                    recent_rewards = env.episode_rewards[-self.log_interval:]
                    recent_users = env.episode_users_served[-self.log_interval:]
                    recent_efficiency = env.episode_power_efficiency[-self.log_interval:]
                    
                    print(f"\nEpisode {self.episode_count}:")
                    print(f"  Reward: {np.mean(recent_rewards):.1f}")
                    print(f"  Users Served: {np.mean(recent_users):.1f}/100")
                    print(f"  Power Efficiency: {np.mean(recent_efficiency):.3f}")
                    
                    # Show improvement
                    if len(env.episode_users_served) >= 20:
                        prev_users = np.mean(env.episode_users_served[-20:-10])
                        curr_users = np.mean(env.episode_users_served[-10:])
                        improvement = curr_users - prev_users
                        print(f"  Improvement: {improvement:+.1f} users")
        
        return True

def plot_demo_results(env, save_path='demo_convergence_results.png'):
    """Create comprehensive results plot"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    episodes = np.arange(1, len(env.episode_rewards) + 1)
    
    # Smooth curves
    window = max(5, len(episodes) // 20)
    
    # 1. Rewards over time
    ax = axes[0, 0]
    if len(episodes) > window:
        rewards_smooth = np.convolve(env.episode_rewards, np.ones(window)/window, mode='valid')
        ax.plot(episodes[:len(rewards_smooth)], rewards_smooth, 'b-', linewidth=2.5)
        ax.fill_between(episodes[:len(rewards_smooth)], 
                       rewards_smooth - np.std(env.episode_rewards[-len(rewards_smooth):]) * 0.3,
                       rewards_smooth + np.std(env.episode_rewards[-len(rewards_smooth):]) * 0.3, 
                       alpha=0.2)
    else:
        ax.plot(episodes, env.episode_rewards, 'b-', linewidth=2.5)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Reward')
    ax.set_title('Reward Learning Curve')
    ax.grid(True, alpha=0.3)
    
    # Add trend line
    if len(episodes) > 10:
        z = np.polyfit(episodes, env.episode_rewards, 1)
        p = np.poly1d(z)
        ax.plot(episodes, p(episodes), "r--", alpha=0.8, linewidth=1.5, label=f'Trend: {z[0]:.2f}')
        ax.legend()
    
    # 2. Users served over time
    ax = axes[0, 1]
    if len(episodes) > window:
        users_smooth = np.convolve(env.episode_users_served, np.ones(window)/window, mode='valid')
        ax.plot(episodes[:len(users_smooth)], users_smooth, 'g-', linewidth=2.5)
        ax.fill_between(episodes[:len(users_smooth)], 
                       users_smooth - np.std(env.episode_users_served[-len(users_smooth):]) * 0.3,
                       users_smooth + np.std(env.episode_users_served[-len(users_smooth):]) * 0.3, 
                       alpha=0.2)
    else:
        ax.plot(episodes, env.episode_users_served, 'g-', linewidth=2.5)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Users Served')
    ax.set_title('Users Served Learning Curve')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    
    # Add improvement annotation
    if len(episodes) > 20:
        start_avg = np.mean(env.episode_users_served[:10])
        end_avg = np.mean(env.episode_users_served[-10:])
        improvement = end_avg - start_avg
        ax.annotate(f'Improvement: +{improvement:.1f} users', 
                   xy=(0.05, 0.95), xycoords='axes fraction',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # 3. Power efficiency over time
    ax = axes[0, 2]
    if len(episodes) > window:
        eff_smooth = np.convolve(env.episode_power_efficiency, np.ones(window)/window, mode='valid')
        ax.plot(episodes[:len(eff_smooth)], eff_smooth, 'r-', linewidth=2.5)
        ax.fill_between(episodes[:len(eff_smooth)], 
                       eff_smooth - np.std(env.episode_power_efficiency[-len(eff_smooth):]) * 0.1,
                       eff_smooth + np.std(env.episode_power_efficiency[-len(eff_smooth):]) * 0.1, 
                       alpha=0.2)
    else:
        ax.plot(episodes, env.episode_power_efficiency, 'r-', linewidth=2.5)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Power Efficiency')
    ax.set_title('Power Efficiency Learning Curve')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # 4. Performance distribution
    ax = axes[1, 0]
    if len(env.episode_users_served) > 50:
        ax.hist(env.episode_users_served[-100:], bins=20, alpha=0.7, color='green', density=True)
        ax.axvline(np.mean(env.episode_users_served[-100:]), color='red', linestyle='--', 
                  linewidth=2, label=f'Mean: {np.mean(env.episode_users_served[-100:]):.1f}')
        ax.set_xlabel('Users Served')
        ax.set_ylabel('Density')
        ax.set_title('Recent Performance Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 5. Convergence analysis
    ax = axes[1, 1]
    if len(episodes) > 30:
        # Rolling standard deviation
        window_std = 20
        rolling_std = []
        for i in range(window_std, len(env.episode_users_served)):
            std_val = np.std(env.episode_users_served[i-window_std:i])
            rolling_std.append(std_val)
        
        x_std = episodes[window_std:]
        ax.plot(x_std, rolling_std, 'purple', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Rolling Std Dev (Users)')
        ax.set_title('Convergence Indicator')
        ax.grid(True, alpha=0.3)
        
        # Add convergence threshold
        threshold = 5
        ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.5, 
                  label=f'Convergence threshold: {threshold}')
        ax.legend()
    
    # 6. Final trajectory visualization
    ax = axes[1, 2]
    if len(env.episode_trajectories) > 0:
        trajectory = env.episode_trajectories[-1]
        
        # Plot trajectory in 2D (top-down view)
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, alpha=0.8, label='UAV Path')
        ax.scatter(env.users_positions[:, 0], env.users_positions[:, 1], 
                  c='orange', s=15, alpha=0.6, label='Users')
        ax.scatter(trajectory[0, 0], trajectory[0, 1], color='green', s=100, 
                  marker='o', label='Start', edgecolors='black')
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], color='red', s=100, 
                  marker='s', label='End', edgecolors='black')
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title('Final Episode Trajectory (Top View)')
        ax.set_xlim(0, 1000)
        ax.set_ylim(0, 1000)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    plt.suptitle('UAV Training Results - Convergence Demonstration', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nDemo results saved to {save_path}")

def main():
    print("UAV PPO Training - Convergence Demonstration")
    print("=" * 50)
    print("Target: Show clear convergence with fewer timesteps")
    print("Optimized for learning speed and convergence detection")
    print("-" * 50)
    
    # Create environment
    env = Monitor(UAVEnvironmentDemo())
    
    # Fast learning hyperparameters
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 128], vf=[256, 128]),
        activation_fn=torch.nn.ReLU,
    )
    
    # Aggressive learning settings
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-3,  # High learning rate
        n_steps=512,  # Smaller batch size for faster updates
        batch_size=64,
        n_epochs=4,
        gamma=0.98,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0,
        tensorboard_log=f"./ppo_demo_tensorboard/PPO_{int(time.time())}"
    )
    
    callback = DemoCallback(env, log_interval=10)
    
    # Train for fewer timesteps but show clear progression
    timesteps = 500_000  # 500 episodes
    print(f"\nTraining for {timesteps:,} timesteps ({timesteps//1000} episodes)...")
    
    start_time = time.time()
    model.learn(total_timesteps=timesteps, callback=callback)
    training_time = time.time() - start_time
    
    print(f"\nTraining completed in {training_time:.1f} seconds")
    
    # Save model
    model_path = f"ppo_demo_converged_{int(time.time())}.zip"
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Generate comprehensive results
    plot_demo_results(env.unwrapped)
    
    # Print final analysis
    env_unwrapped = env.unwrapped
    print("\n" + "="*60)
    print("TRAINING RESULTS ANALYSIS")
    print("="*60)
    
    if len(env_unwrapped.episode_users_served) > 50:
        # Performance progression
        early = env_unwrapped.episode_users_served[:25]
        late = env_unwrapped.episode_users_served[-25:]
        
        early_mean = np.mean(early)
        late_mean = np.mean(late)
        late_std = np.std(late)
        
        print(f"Early performance (first 25 episodes): {early_mean:.1f} users")
        print(f"Late performance (last 25 episodes):  {late_mean:.1f} users")
        print(f"Improvement: +{late_mean - early_mean:.1f} users")
        print(f"Recent stability (std dev): {late_std:.2f}")
        
        # Efficiency progression
        early_eff = np.mean(env_unwrapped.episode_power_efficiency[:25])
        late_eff = np.mean(env_unwrapped.episode_power_efficiency[-25:])
        print(f"\nPower efficiency improvement: {early_eff:.3f} → {late_eff:.3f}")
        
        # Convergence assessment
        if late_std < 5 and late_mean > early_mean + 10:
            print(f"\n✅ CONVERGENCE ACHIEVED!")
            print(f"   Stable performance at ~{late_mean:.0f} users served")
            print(f"   Power efficiency: {late_eff:.3f}")
        elif late_mean > early_mean + 5:
            print(f"\n🔄 GOOD LEARNING PROGRESS")
            print(f"   Model is improving but may need more training for full convergence")
        else:
            print(f"\n⚠️  Limited learning detected")
    
    # Show reward progression
    if len(env_unwrapped.episode_rewards) > 10:
        reward_trend = np.polyfit(range(len(env_unwrapped.episode_rewards)), 
                                 env_unwrapped.episode_rewards, 1)[0]
        print(f"\nReward trend: {reward_trend:+.2f} per episode")
        
        final_reward = np.mean(env_unwrapped.episode_rewards[-10:])
        print(f"Final average reward: {final_reward:.1f}")
    
    print("\nVisualization files generated:")
    print("- demo_convergence_results.png: Complete training analysis")

if __name__ == "__main__":
    main()