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

class UAVEnvironmentImproved(gym.Env):

    def __init__(self):
        super(UAVEnvironmentImproved, self).__init__()
        self.num_users = 100  
        self.num_subchannels = 5  
        self.W_s = 100  
        self.N_0 = (10 ** (-80 / 10)) / 1000  
        self.max_power = 1.0  
        self.R_min = 3.0  
        self.time_granularity = 0.1  
        
        # Constraints
        self.max_acceleration = 2.0  # Increased for better mobility
        self.max_velocity = 5.0  # Increased for better coverage
        self.max_altitude = 1000.0  
        self.min_altitude = 50.0  # Higher min for better coverage
        
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
        
        # Moving averages for smooth learning
        self.users_served_ma = deque(maxlen=100)
        self.power_efficiency_ma = deque(maxlen=100)
        
        self.reset(seed=42)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        reset_position = True
        if options is not None and 'reset_position' in options:
            reset_position = options['reset_position']

        if reset_position:
            # Smart initialization based on learning progress
            if self.episode_count < 100:
                # Early episodes: start from center
                self.uav_position = np.array([500.0, 500.0, 150.0])
            else:
                # Later episodes: start from good positions learned
                self.uav_position = np.array([
                    500.0 + np.random.uniform(-200, 200),
                    500.0 + np.random.uniform(-200, 200),
                    150.0 + np.random.uniform(-50, 50)
                ])
            self.prev_velocity = np.array([0.0, 0.0, 0.0])

        # Generate user distribution - change every 100 episodes
        if reset_position or not hasattr(self, 'users_positions'):
            # Rotate through distributions more gradually
            if self.episode_count < 500:
                distribution_type = 'uniform'
            elif self.episode_count < 1000:
                distribution_type = 'clustered'
            elif self.episode_count < 1500:
                distribution_type = 'edge'
            else:
                distribution_types = ['uniform', 'clustered', 'edge', 'mixed']
                distribution_type = distribution_types[self.episode_count % len(distribution_types)]
            
            self.users_positions = self._generate_user_distribution(distribution_type)
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

    def _generate_user_distribution(self, distribution_type='uniform'):
        if distribution_type == 'uniform':
            return np.random.uniform(0, 1000, size=(self.num_users, 2))
        
        elif distribution_type == 'clustered':
            # Multiple clusters
            n_clusters = np.random.randint(3, 6)
            users_per_cluster = self.num_users // n_clusters
            remaining = self.num_users % n_clusters
            
            all_users = []
            for i in range(n_clusters):
                cluster_center = np.random.uniform(200, 800, size=2)
                cluster_std = np.random.uniform(50, 150)
                n_users = users_per_cluster + (1 if i < remaining else 0)
                cluster_users = np.random.normal(cluster_center, cluster_std, size=(n_users, 2))
                cluster_users = np.clip(cluster_users, 0, 1000)
                all_users.append(cluster_users)
            
            return np.vstack(all_users)
        
        elif distribution_type == 'edge':
            edge_width = 200
            n_per_edge = self.num_users // 4
            
            edge_users = []
            # Left edge
            edge_users.extend([[np.random.uniform(0, edge_width), np.random.uniform(0, 1000)] 
                              for _ in range(n_per_edge)])
            # Right edge
            edge_users.extend([[np.random.uniform(1000-edge_width, 1000), np.random.uniform(0, 1000)] 
                              for _ in range(n_per_edge)])
            # Bottom edge
            edge_users.extend([[np.random.uniform(0, 1000), np.random.uniform(0, edge_width)] 
                              for _ in range(n_per_edge)])
            # Top edge
            edge_users.extend([[np.random.uniform(0, 1000), np.random.uniform(1000-edge_width, 1000)] 
                              for _ in range(self.num_users - 3*n_per_edge)])
            
            return np.array(edge_users)
        
        elif distribution_type == 'mixed':
            n_clustered = self.num_users // 2
            n_uniform = self.num_users - n_clustered
            
            cluster_center = np.random.uniform(200, 800, size=2)
            clustered = np.random.normal(cluster_center, 150, size=(n_clustered, 2))
            clustered = np.clip(clustered, 0, 1000)
            
            uniform = np.random.uniform(0, 1000, size=(n_uniform, 2))
            
            return np.vstack([clustered, uniform])

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

        # Sample and scale acceleration
        a_sample = np.random.beta(a_mean + 1, a_var + 1)
        theta_sample = np.random.beta(theta_mean + 1, theta_var + 1)
        phi_sample = np.random.beta(phi_mean + 1, phi_var + 1)

        a = a_sample * self.max_acceleration  
        theta = theta_sample * 2 * np.pi - np.pi  
        phi = phi_sample * 2 * np.pi - np.pi  

        ax = a * np.sin(theta) * np.cos(phi)
        ay = a * np.sin(theta) * np.sin(phi)
        az = a * np.cos(theta) * 0.5  # Reduced z movement

        acceleration = np.array([ax, ay, az])

        # Update velocity
        new_velocity = self.prev_velocity + self.time_granularity * acceleration
        velocity_magnitude = np.linalg.norm(new_velocity)
        
        # Apply velocity constraint
        if velocity_magnitude > self.max_velocity:
            new_velocity = new_velocity * (self.max_velocity / velocity_magnitude)
        
        self.prev_velocity = new_velocity
        
        # Update position
        new_position = self.uav_position + self.prev_velocity * self.time_granularity
        
        # Apply position constraints
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

        self.power_allocations = self._apply_optimized_power_allocation(
            np.stack([power_means, power_vars], axis=-1))

        # Calculate throughputs
        throughputs = self._calculate_throughputs()
        served_mask = throughputs >= self.R_min
        users_served = np.sum(served_mask)
        
        # Calculate power efficiency
        total_power_to_served = np.sum(self.power_allocations[served_mask])
        total_power_allocated = np.sum(self.power_allocations)
        power_efficiency = total_power_to_served / (total_power_allocated + 1e-8)
        
        # SIMPLIFIED REWARD FUNCTION - Focus on main objectives
        # Primary reward: users served (scaled appropriately)
        reward = users_served * 1.0  # Full reward per user served
        
        # Power efficiency bonus - encourage efficient allocation
        if power_efficiency > 0.7:
            reward += (power_efficiency - 0.7) * 50  # Bonus for high efficiency
        
        # Position penalty - discourage going too high or too low
        if self.uav_position[2] > 300:  # Too high reduces coverage
            reward -= (self.uav_position[2] - 300) * 0.01
        elif self.uav_position[2] < 100:  # Too low limits coverage area
            reward -= (100 - self.uav_position[2]) * 0.01
        
        # Small movement bonus to encourage exploration early on
        if self.episode_count < 200:
            movement = np.linalg.norm(self.prev_velocity[:2])
            if movement > 0.5:
                reward += min(movement * 0.1, 1.0)
        
        # Track metrics
        self.users_served_per_timestep.append(users_served)
        self.throughput_tracker += throughputs
        self.episode_power_efficiency_sum += power_efficiency
        self.timestep_rewards.append(reward)
        self.timestep_users_served.append(users_served)
        self.timestep_power_efficiency.append(power_efficiency)
        
        # Update moving averages
        self.users_served_ma.append(users_served)
        self.power_efficiency_ma.append(power_efficiency)

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
    
    def _apply_optimized_power_allocation(self, power_params):
        """Optimized power allocation focusing on serving maximum users"""
        power_means = power_params[:, :, 0]  
        power_vars = power_params[:, :, 1]  
        
        # Calculate minimum power needed for each user
        min_power_needed = (self.N_0 * (2**(self.R_min/self.W_s) - 1)) / (self.channel_gains + 1e-8)
        
        # Sort users by how easy they are to serve (lowest power needed)
        sorted_users = np.argsort(min_power_needed)
        
        # Initialize allocations
        power_allocations = np.zeros((self.num_users, self.num_subchannels))
        total_power_budget = self.max_power * self.num_users
        power_used = 0
        
        # Greedy allocation: serve easiest users first
        for user in sorted_users:
            if power_used + min_power_needed[user] * 1.2 <= total_power_budget:
                # Allocate 120% of minimum needed for robustness
                user_power = min_power_needed[user] * 1.2
                
                # Distribute across subchannels based on action
                subchannel_weights = power_means[user] + 0.1  # Ensure positive
                subchannel_weights = subchannel_weights / np.sum(subchannel_weights)
                
                power_allocations[user] = subchannel_weights * user_power
                power_used += user_power
            else:
                # Can't serve more users with remaining budget
                break
        
        # Normalize to ensure we don't exceed total power
        if np.sum(power_allocations) > 0:
            power_allocations = power_allocations * (total_power_budget / np.sum(power_allocations))
        
        return power_allocations

    def _get_state(self):
        # Normalize state components
        pos_normalized = self.uav_position / np.array([1000, 1000, self.max_altitude])
        vel_normalized = self.prev_velocity / self.max_velocity
        gains_normalized = self.channel_gains / (np.max(self.channel_gains) + 1e-8)
        
        return np.concatenate([
            pos_normalized,
            vel_normalized,
            gains_normalized
        ]).astype(np.float32)

class TrainingCallback(BaseCallback):
    def __init__(self, env, log_interval=50, verbose=1):
        super(TrainingCallback, self).__init__(verbose)
        self.env = env
        self.log_interval = log_interval
        self.episode_count = 0
        
    def _on_step(self) -> bool:
        # Check if episode ended
        if self.locals.get('dones')[0]:
            self.episode_count += 1
            
            # Log every log_interval episodes
            if self.episode_count % self.log_interval == 0:
                env = self.env.unwrapped
                recent_rewards = env.episode_rewards[-self.log_interval:] if len(env.episode_rewards) >= self.log_interval else env.episode_rewards
                recent_users = env.episode_users_served[-self.log_interval:] if len(env.episode_users_served) >= self.log_interval else env.episode_users_served
                recent_efficiency = env.episode_power_efficiency[-self.log_interval:] if len(env.episode_power_efficiency) >= self.log_interval else env.episode_power_efficiency
                
                print(f"\nEpisode {self.episode_count}:")
                print(f"  Avg Reward: {np.mean(recent_rewards):.2f}")
                print(f"  Avg Users Served: {np.mean(recent_users):.2f}")
                print(f"  Avg Power Efficiency: {np.mean(recent_efficiency):.3f}")
                print(f"  Learning Rate: {self.model.learning_rate}")
        
        return True

def plot_training_results(env, save_path='training_results_improved.png'):
    """Plot training curves without target lines"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    episodes = np.arange(1, len(env.episode_rewards) + 1)
    
    # Smooth curves using moving average
    window = 50
    
    # Rewards
    ax = axes[0, 0]
    rewards_smooth = np.convolve(env.episode_rewards, np.ones(window)/window, mode='valid')
    ax.plot(episodes[:len(rewards_smooth)], rewards_smooth, 'b-', alpha=0.8)
    ax.fill_between(episodes[:len(rewards_smooth)], rewards_smooth - np.std(env.episode_rewards) * 0.5,
                    rewards_smooth + np.std(env.episode_rewards) * 0.5, alpha=0.3)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Reward')
    ax.set_title('Training Rewards')
    ax.grid(True, alpha=0.3)
    
    # Users Served
    ax = axes[0, 1]
    users_smooth = np.convolve(env.episode_users_served, np.ones(window)/window, mode='valid')
    ax.plot(episodes[:len(users_smooth)], users_smooth, 'g-', alpha=0.8)
    ax.fill_between(episodes[:len(users_smooth)], users_smooth - np.std(env.episode_users_served) * 0.5,
                    users_smooth + np.std(env.episode_users_served) * 0.5, alpha=0.3)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Users Served')
    ax.set_title('Users Served per Episode')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    
    # Power Efficiency
    ax = axes[1, 0]
    efficiency_smooth = np.convolve(env.episode_power_efficiency, np.ones(window)/window, mode='valid')
    ax.plot(episodes[:len(efficiency_smooth)], efficiency_smooth, 'r-', alpha=0.8)
    ax.fill_between(episodes[:len(efficiency_smooth)], efficiency_smooth - np.std(env.episode_power_efficiency) * 0.1,
                    efficiency_smooth + np.std(env.episode_power_efficiency) * 0.1, alpha=0.3)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Power Efficiency')
    ax.set_title('Power Allocation Efficiency')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # Combined metrics
    ax = axes[1, 1]
    ax.plot(episodes[:len(users_smooth)], users_smooth / 100, 'g-', label='Users Served %', alpha=0.8)
    ax.plot(episodes[:len(efficiency_smooth)], efficiency_smooth, 'r-', label='Power Efficiency', alpha=0.8)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Normalized Metrics')
    ax.set_title('Combined Performance Metrics')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training results saved to {save_path}")

def test_on_distributions(model, env_class, save_path='test_results_improved.png'):
    """Test trained model on different user distributions"""
    distributions = ['uniform', 'clustered', 'edge', 'mixed']
    results = {dist: {'users_served': [], 'power_efficiency': []} for dist in distributions}
    
    for dist in distributions:
        print(f"\nTesting on {dist} distribution...")
        env = env_class()
        
        for episode in range(10):  # Test 10 episodes per distribution
            obs, _ = env.reset(options={'user_distribution': dist})
            episode_users = []
            episode_efficiency = []
            
            for _ in range(1000):  # Run full episode
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_users.append(info['users_served'])
                episode_efficiency.append(info['power_efficiency'])
                
                if terminated or truncated:
                    break
            
            results[dist]['users_served'].append(np.mean(episode_users))
            results[dist]['power_efficiency'].append(np.mean(episode_efficiency))
    
    # Plot test results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Users served
    dist_names = list(results.keys())
    users_means = [np.mean(results[d]['users_served']) for d in dist_names]
    users_stds = [np.std(results[d]['users_served']) for d in dist_names]
    
    ax1.bar(dist_names, users_means, yerr=users_stds, capsize=10, alpha=0.7, color='green')
    ax1.set_ylabel('Average Users Served')
    ax1.set_title('Performance on Different User Distributions')
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)
    
    # Power efficiency
    efficiency_means = [np.mean(results[d]['power_efficiency']) for d in dist_names]
    efficiency_stds = [np.std(results[d]['power_efficiency']) for d in dist_names]
    
    ax2.bar(dist_names, efficiency_means, yerr=efficiency_stds, capsize=10, alpha=0.7, color='red')
    ax2.set_ylabel('Average Power Efficiency')
    ax2.set_title('Power Efficiency on Different Distributions')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Test results saved to {save_path}")
    
    # Print summary
    for dist in dist_names:
        print(f"\n{dist.capitalize()} distribution:")
        print(f"  Users Served: {np.mean(results[dist]['users_served']):.1f} ± {np.std(results[dist]['users_served']):.1f}")
        print(f"  Power Efficiency: {np.mean(results[dist]['power_efficiency']):.3f} ± {np.std(results[dist]['power_efficiency']):.3f}")

def visualize_trajectory(env, episode_idx=-1, save_path='trajectory_improved.png'):
    """Visualize UAV trajectory for a specific episode"""
    if episode_idx == -1:
        trajectory = env.episode_trajectories[-1]
        episode_num = len(env.episode_trajectories)
    else:
        trajectory = env.episode_trajectories[episode_idx]
        episode_num = episode_idx + 1
    
    trajectory = np.array(trajectory)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
            'b-', linewidth=2, alpha=0.8, label='UAV Path')
    
    # Plot start and end points
    ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], 
               color='green', s=200, marker='o', label='Start', edgecolors='black')
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], 
               color='red', s=200, marker='s', label='End', edgecolors='black')
    
    # Plot users
    ax.scatter(env.users_positions[:, 0], env.users_positions[:, 1], env.users_positions[:, 2], 
               color='orange', s=30, alpha=0.6, label='Users')
    
    # Add trajectory progress colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(trajectory)))
    for i in range(len(trajectory) - 1):
        ax.plot(trajectory[i:i+2, 0], trajectory[i:i+2, 1], trajectory[i:i+2, 2],
                color=colors[i], linewidth=3, alpha=0.7)
    
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Altitude (m)')
    ax.set_title(f'UAV Trajectory - Episode {episode_num}')
    ax.legend()
    
    # Set view angle
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Trajectory visualization saved to {save_path}")

def main():
    print("Starting improved UAV training with 2M timesteps...")
    
    # Create environment
    env = Monitor(UAVEnvironmentImproved())
    
    # Define optimized hyperparameters
    policy_kwargs = dict(
        net_arch=dict(pi=[512, 512, 256], vf=[512, 512, 256]),  # Separate networks
        activation_fn=torch.nn.ReLU,
    )
    
    # Create PPO model with tuned hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=lambda f: f * 3e-4 + (1 - f) * 1e-5,  # Learning rate schedule
        n_steps=2048,  # More steps before update
        batch_size=64,  # Smaller batches for stability
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.01,  # Exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0,
        tensorboard_log=f"./ppo_improved_tensorboard/PPO_{int(time.time())}"
    )
    
    # Create callback
    callback = TrainingCallback(env, log_interval=50)
    
    # Train for 2M timesteps (2000 episodes)
    print("\nTraining for 2,000,000 timesteps (2000 episodes)...")
    model.learn(total_timesteps=2_000_000, callback=callback, progress_bar=True)
    
    # Save model
    model_path = f"ppo_improved_model_{int(time.time())}.zip"
    model.save(model_path)
    print(f"\nModel saved to {model_path}")
    
    # Generate plots
    print("\nGenerating training plots...")
    plot_training_results(env.unwrapped)
    
    # Test on different distributions
    print("\nTesting on different user distributions...")
    test_on_distributions(model, UAVEnvironmentImproved)
    
    # Visualize final trajectory
    print("\nVisualizing final trajectory...")
    visualize_trajectory(env.unwrapped)
    
    # Print final statistics
    env_unwrapped = env.unwrapped
    print("\n" + "="*50)
    print("FINAL TRAINING STATISTICS")
    print("="*50)
    print(f"Total Episodes: {len(env_unwrapped.episode_rewards)}")
    print(f"Final Avg Users Served: {np.mean(env_unwrapped.episode_users_served[-100:]):.2f}")
    print(f"Final Avg Power Efficiency: {np.mean(env_unwrapped.episode_power_efficiency[-100:]):.3f}")
    print(f"Final Avg Reward: {np.mean(env_unwrapped.episode_rewards[-100:]):.2f}")
    
    # Check convergence
    if len(env_unwrapped.episode_users_served) > 200:
        recent = env_unwrapped.episode_users_served[-100:]
        older = env_unwrapped.episode_users_served[-200:-100]
        improvement = np.mean(recent) - np.mean(older)
        print(f"Improvement over last 100 episodes: {improvement:.2f} users")
        
        if abs(improvement) < 2 and np.std(recent) < 5:
            print("✓ Model has converged!")
        else:
            print("⚠ Model may need more training for full convergence")

if __name__ == "__main__":
    main()