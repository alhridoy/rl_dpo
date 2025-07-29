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

class UAVEnvironmentFinal(gym.Env):

    def __init__(self):
        super(UAVEnvironmentFinal, self).__init__()
        self.num_users = 100  
        self.num_subchannels = 5  
        self.W_s = 100  
        self.N_0 = (10 ** (-80 / 10)) / 1000  
        self.max_power = 1.0  
        self.R_min = 3.0  
        self.time_granularity = 0.1  
        
        # Constraints
        self.max_acceleration = 2.0
        self.max_velocity = 5.0
        self.max_altitude = 300.0
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
        
        self.reset(seed=42)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        # Always reset UAV to center
        self.uav_position = np.array([500.0, 500.0, 150.0])
        self.prev_velocity = np.array([0.0, 0.0, 0.0])

        # Generate user distribution - gradually increase complexity
        if self.episode_count < 100:
            # Simple clustered distribution for initial learning
            center = np.array([500, 500])
            std = 150
            self.users_positions = np.random.normal(center, std, size=(self.num_users, 2))
        elif self.episode_count < 300:
            # Two clusters
            n1 = self.num_users // 2
            n2 = self.num_users - n1
            c1 = np.array([400, 400])
            c2 = np.array([600, 600])
            cluster1 = np.random.normal(c1, 120, size=(n1, 2))
            cluster2 = np.random.normal(c2, 120, size=(n2, 2))
            self.users_positions = np.vstack([cluster1, cluster2])
        elif self.episode_count < 600:
            # Three clusters
            n_per = self.num_users // 3
            centers = [[300, 300], [700, 300], [500, 700]]
            clusters = []
            for i, center in enumerate(centers):
                n = n_per if i < 2 else self.num_users - 2*n_per
                cluster = np.random.normal(center, 100, size=(n, 2))
                clusters.append(cluster)
            self.users_positions = np.vstack(clusters)
        else:
            # Mixed patterns for final training
            pattern = (self.episode_count // 50) % 4
            if pattern == 0:
                # Uniform
                self.users_positions = np.random.uniform(100, 900, size=(self.num_users, 2))
            elif pattern == 1:
                # Ring
                angles = np.linspace(0, 2*np.pi, self.num_users)
                radius = 250 + np.random.normal(0, 50, self.num_users)
                self.users_positions = np.column_stack([
                    500 + radius * np.cos(angles),
                    500 + radius * np.sin(angles)
                ])
            elif pattern == 2:
                # Four corners
                n_per = self.num_users // 4
                corners = [[200, 200], [800, 200], [200, 800], [800, 800]]
                clusters = []
                for i, corner in enumerate(corners):
                    n = n_per if i < 3 else self.num_users - 3*n_per
                    cluster = np.random.normal(corner, 80, size=(n, 2))
                    clusters.append(cluster)
                self.users_positions = np.vstack(clusters)
            else:
                # Edge distribution
                edge_width = 150
                n_per_edge = self.num_users // 4
                edges = []
                # Left edge
                edges.extend([[np.random.uniform(0, edge_width), np.random.uniform(0, 1000)] 
                             for _ in range(n_per_edge)])
                # Right edge  
                edges.extend([[np.random.uniform(1000-edge_width, 1000), np.random.uniform(0, 1000)] 
                             for _ in range(n_per_edge)])
                # Bottom edge
                edges.extend([[np.random.uniform(0, 1000), np.random.uniform(0, edge_width)] 
                             for _ in range(n_per_edge)])
                # Top edge
                edges.extend([[np.random.uniform(0, 1000), np.random.uniform(1000-edge_width, 1000)] 
                             for _ in range(self.num_users - 3*n_per_edge)])
                self.users_positions = np.array(edges)
        
        # Clip to bounds
        self.users_positions = np.clip(self.users_positions, 50, 950)
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
        # Extract movement parameters
        a_mean, a_var = action[0], action[1]
        theta_mean, theta_var = action[2], action[3]
        phi_mean, phi_var = action[4], action[5]

        # Simple deterministic movement for consistent learning
        a = a_mean * self.max_acceleration  
        theta = theta_mean * 2 * np.pi - np.pi  
        phi = phi_mean * np.pi/3 - np.pi/6  # Limited vertical range

        ax = a * np.sin(theta) * np.cos(phi)
        ay = a * np.sin(theta) * np.sin(phi)
        az = a * np.cos(theta) * 0.2  # Minimal vertical movement

        acceleration = np.array([ax, ay, az])

        # Update velocity and position
        new_velocity = self.prev_velocity + self.time_granularity * acceleration
        velocity_magnitude = np.linalg.norm(new_velocity)
        
        if velocity_magnitude > self.max_velocity:
            new_velocity = new_velocity * (self.max_velocity / velocity_magnitude)
        
        self.prev_velocity = new_velocity
        
        new_position = self.uav_position + self.prev_velocity * self.time_granularity
        new_position[0] = np.clip(new_position[0], 50, 950)
        new_position[1] = np.clip(new_position[1], 50, 950)
        new_position[2] = np.clip(new_position[2], self.min_altitude, self.max_altitude)
        
        self.uav_position = new_position
        self.trajectory.append(self.uav_position.copy())
        
        # Update channel gains
        self._update_channel_gains()
        
        # Power allocation
        power_means = action[6: 6 + self.num_users * self.num_subchannels]
        power_vars = action[6 + self.num_users * self.num_subchannels: ]  

        power_means = power_means.reshape(self.num_users, self.num_subchannels)
        power_vars = power_vars.reshape(self.num_users, self.num_subchannels)

        self.power_allocations = self._apply_greedy_power_allocation(power_means)

        # Calculate performance
        throughputs = self._calculate_throughputs()
        served_mask = throughputs >= self.R_min
        users_served = np.sum(served_mask)
        
        # Calculate power efficiency
        total_power_to_served = np.sum(self.power_allocations[served_mask]) if np.any(served_mask) else 0
        total_power_allocated = np.sum(self.power_allocations)
        power_efficiency = total_power_to_served / (total_power_allocated + 1e-8) if total_power_allocated > 0 else 0
        
        # CLEAN REWARD FUNCTION - ONLY USERS SERVED MATTERS
        reward = float(users_served)  # Pure reward = number of users served
        
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
    
    def _apply_greedy_power_allocation(self, power_means):
        """Greedy power allocation guided by neural network output"""
        # Calculate minimum power needed for each user
        min_power_needed = (self.N_0 * (2**(self.R_min/self.W_s) - 1)) / (self.channel_gains + 1e-8)
        
        # Use neural network output to weight user priorities
        user_priorities = np.mean(power_means, axis=1)  # Average across subchannels
        
        # Sort users by combined priority (channel quality + network preference)
        combined_score = self.channel_gains * (1 + user_priorities)
        sorted_indices = np.argsort(combined_score)[::-1]
        
        # Allocate power greedily
        power_allocations = np.zeros((self.num_users, self.num_subchannels))
        total_budget = self.max_power * self.num_users * 0.95
        used_budget = 0
        
        for user_idx in sorted_indices:
            required_power = min_power_needed[user_idx] * 1.05  # 5% safety margin
            
            if used_budget + required_power <= total_budget:
                # Distribute power across subchannels based on network output
                subchannel_weights = power_means[user_idx] + 0.1  # Ensure positive
                subchannel_weights = subchannel_weights / np.sum(subchannel_weights)
                
                power_allocations[user_idx] = subchannel_weights * required_power
                used_budget += required_power
            else:
                # Try to allocate remaining budget
                remaining = total_budget - used_budget
                if remaining > min_power_needed[user_idx] * 0.8:
                    subchannel_weights = power_means[user_idx] + 0.1
                    subchannel_weights = subchannel_weights / np.sum(subchannel_weights)
                    power_allocations[user_idx] = subchannel_weights * remaining
                    used_budget = total_budget
                break
        
        return power_allocations

    def _get_state(self):
        # Normalize state components
        pos_normalized = self.uav_position / np.array([1000, 1000, self.max_altitude])
        vel_normalized = np.clip(self.prev_velocity / self.max_velocity, -1, 1)
        gains_normalized = self.channel_gains / (np.max(self.channel_gains) + 1e-8)
        
        return np.concatenate([
            pos_normalized,
            vel_normalized,
            gains_normalized
        ]).astype(np.float32)

class ProgressCallback(BaseCallback):
    def __init__(self, env, log_interval=50, verbose=1):
        super(ProgressCallback, self).__init__(verbose)
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
                    print(f"  Avg Reward: {np.mean(recent_rewards):.2f}")
                    print(f"  Avg Users Served: {np.mean(recent_users):.2f}/100")
                    print(f"  Avg Power Efficiency: {np.mean(recent_efficiency):.3f}")
                    
                    # Check for improvement
                    if len(env.episode_users_served) >= 100:
                        prev_100 = np.mean(env.episode_users_served[-100:-50])
                        recent_50 = np.mean(env.episode_users_served[-50:])
                        improvement = recent_50 - prev_100
                        print(f"  Recent improvement: {improvement:+.2f} users")
                        
                        # Check convergence
                        variance = np.var(env.episode_users_served[-50:])
                        if variance < 5 and recent_50 > 80:
                            print(f"  🎯 Approaching convergence! (variance: {variance:.2f})")
        
        return True

def plot_clean_results(env, save_path='final_convergence_results.png'):
    """Plot clean results without any target lines"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    episodes = np.arange(1, len(env.episode_rewards) + 1)
    
    # Use appropriate smoothing
    window = max(20, len(episodes) // 50)
    
    # 1. Rewards - should increase and converge
    ax = axes[0, 0]
    if len(episodes) > window:
        rewards_smooth = np.convolve(env.episode_rewards, np.ones(window)/window, mode='valid')
        episodes_smooth = episodes[:len(rewards_smooth)]
        ax.plot(episodes_smooth, rewards_smooth, 'b-', linewidth=2.5)
        ax.fill_between(episodes_smooth, 
                       rewards_smooth - np.std(env.episode_rewards[-len(rewards_smooth):]) * 0.2,
                       rewards_smooth + np.std(env.episode_rewards[-len(rewards_smooth):]) * 0.2, 
                       alpha=0.3, color='blue')
    else:
        ax.plot(episodes, env.episode_rewards, 'b-', linewidth=2.5)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Reward (Users Served)')
    ax.set_title('Reward Learning Curve')
    ax.grid(True, alpha=0.3)
    
    # 2. Users Served - should increase and converge
    ax = axes[0, 1]
    if len(episodes) > window:
        users_smooth = np.convolve(env.episode_users_served, np.ones(window)/window, mode='valid')
        episodes_smooth = episodes[:len(users_smooth)]
        ax.plot(episodes_smooth, users_smooth, 'g-', linewidth=2.5)
        ax.fill_between(episodes_smooth, 
                       users_smooth - np.std(env.episode_users_served[-len(users_smooth):]) * 0.2,
                       users_smooth + np.std(env.episode_users_served[-len(users_smooth):]) * 0.2, 
                       alpha=0.3, color='green')
    else:
        ax.plot(episodes, env.episode_users_served, 'g-', linewidth=2.5)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Users Served')
    ax.set_title('Users Served Learning Curve')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    
    # 3. Power Efficiency - should increase and converge
    ax = axes[1, 0]
    if len(episodes) > window:
        eff_smooth = np.convolve(env.episode_power_efficiency, np.ones(window)/window, mode='valid')
        episodes_smooth = episodes[:len(eff_smooth)]
        ax.plot(episodes_smooth, eff_smooth, 'r-', linewidth=2.5)
        ax.fill_between(episodes_smooth, 
                       eff_smooth - np.std(env.episode_power_efficiency[-len(eff_smooth):]) * 0.1,
                       eff_smooth + np.std(env.episode_power_efficiency[-len(eff_smooth):]) * 0.1, 
                       alpha=0.3, color='red')
    else:
        ax.plot(episodes, env.episode_power_efficiency, 'r-', linewidth=2.5)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Power Efficiency')
    ax.set_title('Power Efficiency Learning Curve')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # 4. Final trajectory
    ax = axes[1, 1]
    if len(env.episode_trajectories) > 0:
        trajectory = env.episode_trajectories[-1]
        
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, alpha=0.8, label='UAV Path')
        ax.scatter(env.users_positions[:, 0], env.users_positions[:, 1], 
                  c='orange', s=15, alpha=0.6, label=f'Users ({len(env.users_positions)})')
        ax.scatter(trajectory[0, 0], trajectory[0, 1], color='green', s=100, 
                  marker='o', label='Start', edgecolors='black')
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], color='red', s=100, 
                  marker='s', label='End', edgecolors='black')
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title('Final Episode Trajectory')
        ax.set_xlim(0, 1000)
        ax.set_ylim(0, 1000)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    plt.suptitle('UAV Training Results - Convergence Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Clean results saved to {save_path}")

def test_distributions(model, env_class, save_path='distribution_test_results.png'):
    """Test on different user distributions"""
    distributions = ['clustered', 'uniform', 'ring', 'edge']
    results = {}
    
    for dist in distributions:
        print(f"Testing on {dist} distribution...")
        env = env_class()
        
        # Force specific distribution
        if dist == 'clustered':
            center = np.array([500, 500])
            env.users_positions = np.random.normal(center, 150, size=(100, 2))
        elif dist == 'uniform':
            env.users_positions = np.random.uniform(100, 900, size=(100, 2))
        elif dist == 'ring':
            angles = np.linspace(0, 2*np.pi, 100)
            radius = 250
            env.users_positions = np.column_stack([
                500 + radius * np.cos(angles),
                500 + radius * np.sin(angles)
            ])
        elif dist == 'edge':
            edge_users = []
            n_per_edge = 25
            edge_users.extend([[np.random.uniform(0, 150), np.random.uniform(0, 1000)] for _ in range(n_per_edge)])
            edge_users.extend([[np.random.uniform(850, 1000), np.random.uniform(0, 1000)] for _ in range(n_per_edge)])
            edge_users.extend([[np.random.uniform(0, 1000), np.random.uniform(0, 150)] for _ in range(n_per_edge)])
            edge_users.extend([[np.random.uniform(0, 1000), np.random.uniform(850, 1000)] for _ in range(25)])
            env.users_positions = np.array(edge_users)
        
        env.users_positions = np.clip(env.users_positions, 50, 950)
        env.users_positions = np.column_stack((env.users_positions, np.zeros(100)))
        
        # Test multiple episodes
        episode_results = []
        for _ in range(10):
            obs, _ = env.reset()
            episode_users = []
            
            for _ in range(1000):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_users.append(info['users_served'])
                
                if terminated or truncated:
                    break
            
            episode_results.append(np.mean(episode_users))
        
        results[dist] = episode_results
    
    # Plot test results
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    dist_names = list(results.keys())
    means = [np.mean(results[d]) for d in dist_names]
    stds = [np.std(results[d]) for d in dist_names]
    
    bars = ax.bar(dist_names, means, yerr=stds, capsize=10, alpha=0.7, 
                  color=['green', 'blue', 'red', 'orange'])
    
    ax.set_ylabel('Average Users Served')
    ax.set_title('Performance on Different User Distributions')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 1,
                f'{mean:.1f}±{std:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Distribution test results saved to {save_path}")
    
    # Print results
    print("\nTest Results Summary:")
    for dist in dist_names:
        mean_perf = np.mean(results[dist])
        std_perf = np.std(results[dist])
        print(f"  {dist.capitalize()}: {mean_perf:.1f} ± {std_perf:.1f} users served")

def main():
    print("UAV Final Training - 1M Timesteps")
    print("=" * 50)
    print("Objective: Clear convergence with increasing users served")
    print("Reward: ONLY number of users served (no other components)")
    print("-" * 50)
    
    # Create environment
    env = Monitor(UAVEnvironmentFinal())
    
    # Optimized hyperparameters for convergence
    policy_kwargs = dict(
        net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]),
        activation_fn=torch.nn.ReLU,
    )
    
    # PPO with settings optimized for this problem
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,  # Standard learning rate
        n_steps=2048,  # Collect more experience before updating
        batch_size=64,  # Smaller batches for stable learning
        n_epochs=10,  # More epochs per update
        gamma=0.99,  # High gamma for long-term planning
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Some exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0,
        tensorboard_log=f"./final_tensorboard/PPO_{int(time.time())}"
    )
    
    # Training callback
    callback = ProgressCallback(env, log_interval=25)
    
    # Train for 1M timesteps (1000 episodes)
    print("\nStarting training for 1,000,000 timesteps (1000 episodes)...")
    print("This should take 15-30 minutes depending on hardware...")
    
    start_time = time.time()
    model.learn(total_timesteps=1_000_000, callback=callback)
    training_time = time.time() - start_time
    
    print(f"\nTraining completed in {training_time/60:.1f} minutes")
    
    # Save model
    model_path = f"uav_final_model_{int(time.time())}.zip"
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Generate clean plots
    print("\nGenerating convergence plots...")
    plot_clean_results(env.unwrapped)
    
    # Test on different distributions
    print("\nTesting on different user distributions...")
    test_distributions(model, UAVEnvironmentFinal)
    
    # Final analysis
    env_unwrapped = env.unwrapped
    print("\n" + "="*60)
    print("FINAL TRAINING ANALYSIS")
    print("="*60)
    
    if len(env_unwrapped.episode_rewards) >= 100:
        # Performance progression
        early_episodes = env_unwrapped.episode_users_served[:100]
        late_episodes = env_unwrapped.episode_users_served[-100:]
        
        early_mean = np.mean(early_episodes)
        late_mean = np.mean(late_episodes)
        late_std = np.std(late_episodes)
        
        print(f"Early performance (episodes 1-100):   {early_mean:.2f} users served")
        print(f"Late performance (episodes 901-1000): {late_mean:.2f} users served")
        print(f"Improvement: {late_mean - early_mean:+.2f} users")
        print(f"Final stability (std dev): {late_std:.2f}")
        
        # Power efficiency
        early_eff = np.mean(env_unwrapped.episode_power_efficiency[:100])
        late_eff = np.mean(env_unwrapped.episode_power_efficiency[-100:])
        print(f"\nPower efficiency:")
        print(f"  Early: {early_eff:.3f}")
        print(f"  Final: {late_eff:.3f}")
        print(f"  Improvement: {late_eff - early_eff:+.3f}")
        
        # Convergence assessment
        if late_std < 3 and late_mean > early_mean + 10:
            print(f"\n✅ EXCELLENT CONVERGENCE ACHIEVED!")
            print(f"   Stable at {late_mean:.0f} users served")
            print(f"   Low variance indicates convergence")
        elif late_mean > early_mean + 5:
            print(f"\n🔄 GOOD LEARNING PROGRESS")
            print(f"   Model improved significantly")
        else:
            print(f"\n⚠️  Limited improvement - may need more training")
        
        # Check if reward correlates with users served
        reward_correlation = np.corrcoef(env_unwrapped.episode_rewards, env_unwrapped.episode_users_served)[0,1]
        print(f"\nReward-Users correlation: {reward_correlation:.3f}")
        if reward_correlation > 0.95:
            print("✅ Reward properly tracks users served")
        else:
            print("⚠️  Reward and users served not well correlated")
    
    print(f"\nFiles generated:")
    print(f"  - final_convergence_results.png: Training curves")
    print(f"  - distribution_test_results.png: Performance on different distributions")
    print(f"  - {model_path}: Trained model")

if __name__ == "__main__":
    main()