import numpy as np
import matplotlib.pyplot as plt
import time
from collections import deque
import os

class UAVEnvironmentDemo:
    def __init__(self):
        self.num_users = 100  
        self.num_subchannels = 5  
        self.W_s = 100  
        self.N_0 = (10 ** (-80 / 10)) / 1000  
        self.max_power = 1.0  
        self.R_min = 3.0  
        self.time_granularity = 0.1  
        
        # Constraints
        self.max_acceleration = 1.0
        self.max_velocity = 3.0
        self.max_altitude = 1000.0
        self.min_altitude = 10.0
        
        # Penalty weights
        self.altitude_penalty_weight = 10.0
        self.velocity_penalty_weight = 5.0
        self.power_waste_penalty_weight = 2.0
        
        # Exploration parameters
        self.exploration_bonus = 5.0
        self.position_repeat_penalty = 10.0
        
        self.uav_position = np.array([500, 500, 100])  
        self.prev_velocity = np.array([0.0, 0.0, 0.0])
        self.position_history = deque(maxlen=100)
        
        # Episode tracking
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_users_served = []
        self.episode_power_efficiency = []
        self.episode_trajectories = []
        
        self.reset()

    def reset(self):
        # Add randomization to starting position
        self.uav_position = np.array([
            500.0 + np.random.uniform(-100, 100),
            500.0 + np.random.uniform(-100, 100),
            100.0 + np.random.uniform(-20, 20)
        ])
        self.prev_velocity = np.array([0.0, 0.0, 0.0])
        self.position_history = deque(maxlen=100)
        
        # Generate user distribution
        distribution_types = ['uniform', 'clustered', 'edge', 'mixed']
        distribution_type = distribution_types[self.episode_count % len(distribution_types)]
        self.users_positions = self._generate_user_distribution(distribution_type)
        self.users_positions = np.column_stack((self.users_positions, np.zeros(self.num_users)))
        
        self._update_channel_gains()
        
        self.step_count = 0
        self.trajectory = [self.uav_position.copy()]
        
        return self._get_state()
    
    def _generate_user_distribution(self, distribution_type):
        if distribution_type == 'uniform':
            return np.random.uniform(0, 1000, size=(self.num_users, 2))
        elif distribution_type == 'clustered':
            clusters = []
            n_clusters = 3
            cluster_centers = np.array([[200, 200], [800, 800], [200, 800]])
            users_per_cluster = self.num_users // n_clusters
            
            for i, center in enumerate(cluster_centers):
                remaining = self.num_users - i * users_per_cluster if i == n_clusters - 1 else users_per_cluster
                cluster = np.random.normal(center, 100, size=(remaining, 2))
                clusters.append(np.clip(cluster, 0, 1000))
            
            return np.vstack(clusters)
        elif distribution_type == 'edge':
            edge_users = []
            edge_width = 100
            n_per_edge = self.num_users // 4
            
            for _ in range(n_per_edge):
                edge_users.append([np.random.uniform(0, edge_width), np.random.uniform(0, 1000)])
            for _ in range(n_per_edge):
                edge_users.append([np.random.uniform(1000-edge_width, 1000), np.random.uniform(0, 1000)])
            for _ in range(n_per_edge):
                edge_users.append([np.random.uniform(0, 1000), np.random.uniform(0, edge_width)])
            for _ in range(self.num_users - 3*n_per_edge):
                edge_users.append([np.random.uniform(0, 1000), np.random.uniform(1000-edge_width, 1000)])
            
            return np.array(edge_users)
        else:  # mixed
            n_clustered = self.num_users // 2
            cluster_center = np.random.uniform(200, 800, size=2)
            clustered = np.random.normal(cluster_center, 150, size=(n_clustered, 2))
            clustered = np.clip(clustered, 0, 1000)
            uniform = np.random.uniform(0, 1000, size=(self.num_users - n_clustered, 2))
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

    def step(self, action=None):
        # Smart movement policy based on user distribution
        if action is None:
            # Find underserved areas
            throughputs = self._calculate_throughputs()
            served_mask = throughputs >= self.R_min
            
            if np.any(~served_mask):
                # Move towards unserved users
                unserved_positions = self.users_positions[~served_mask, :2]
                target = unserved_positions[np.random.randint(len(unserved_positions))]
                
                # Calculate direction
                direction = target - self.uav_position[:2]
                direction = direction / (np.linalg.norm(direction) + 1e-8)
                
                # Add some randomness for exploration
                direction += np.random.normal(0, 0.1, size=2)
                direction = direction / (np.linalg.norm(direction) + 1e-8)
                
                # Set acceleration
                ax = direction[0] * 0.8
                ay = direction[1] * 0.8
                az = np.random.uniform(-0.1, 0.1)  # Slight altitude variation
            else:
                # Explore new areas
                ax = np.random.uniform(-0.5, 0.5)
                ay = np.random.uniform(-0.5, 0.5)
                az = np.random.uniform(-0.1, 0.1)
        
        acceleration = np.array([ax, ay, az])
        
        # Update velocity with constraints
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
        self.position_history.append(self.uav_position[:2].copy())
        
        # Update channel gains
        self._update_channel_gains()
        
        # Smart power allocation
        self.power_allocations = self._apply_smart_power_allocation()
        
        # Calculate metrics
        throughputs = self._calculate_throughputs()
        served_mask = throughputs >= self.R_min
        users_served = np.sum(served_mask)
        
        total_power_to_served = np.sum(self.power_allocations[served_mask])
        total_power_to_unserved = np.sum(self.power_allocations[~served_mask])
        total_power_allocated = np.sum(self.power_allocations)
        
        power_efficiency = total_power_to_served / (total_power_allocated + 1e-8)
        
        # Calculate reward
        reward = users_served * 0.5
        reward -= self.power_waste_penalty_weight * (total_power_to_unserved / (self.max_power * self.num_users + 1e-8)) * 200
        
        movement_magnitude = np.linalg.norm(self.prev_velocity[:2])
        if movement_magnitude > 0.5:
            reward += min(movement_magnitude * 3, 10)
        elif movement_magnitude < 0.1:
            reward -= 5
        
        self.step_count += 1
        terminated = self.step_count >= 1000
        
        return self._get_state(), reward, terminated, {
            'users_served': users_served,
            'power_efficiency': power_efficiency,
            'velocity_magnitude': velocity_magnitude,
            'altitude': self.uav_position[2]
        }

    def _calculate_throughputs(self):
        throughputs = np.zeros(self.num_users)
        for user in range(self.num_users):
            total_power = np.sum(self.power_allocations[user]) if hasattr(self, 'power_allocations') else self.max_power
            snr = (self.channel_gains[user] * total_power) / self.N_0
            throughputs[user] = self.W_s * np.log2(1 + snr)
        return throughputs
    
    def _apply_smart_power_allocation(self):
        # Calculate channel quality
        channel_quality = self.channel_gains / np.max(self.channel_gains + 1e-8)
        
        # Calculate minimum power needed
        min_power_needed = (self.N_0 * (2**(self.R_min/self.W_s) - 1)) / (self.channel_gains + 1e-8)
        
        # Identify servable users
        servable_mask = min_power_needed < self.max_power * 0.5
        
        # Initialize allocations
        power_allocations = np.zeros((self.num_users, self.num_subchannels))
        
        # Allocate to servable users
        servable_indices = np.where(servable_mask)[0]
        if len(servable_indices) > 0:
            sorted_servable = servable_indices[np.argsort(channel_quality[servable_indices])[::-1]]
            total_power_budget = self.max_power * self.num_users * 0.9
            
            for idx, user in enumerate(sorted_servable):
                user_power_budget = min(
                    min_power_needed[user] * 1.5,
                    total_power_budget / (idx + 1)
                )
                
                if user_power_budget > 0.01:
                    power_allocations[user] = np.ones(self.num_subchannels) * user_power_budget / self.num_subchannels
                    total_power_budget -= user_power_budget
        
        # Minimal power to unservable users
        unservable_indices = np.where(~servable_mask)[0]
        for user in unservable_indices:
            power_allocations[user] = np.ones(self.num_subchannels) * 0.001 / self.num_subchannels
        
        return power_allocations

    def _get_state(self):
        epsilon = 1e-8
        max_pos = 1000.0
        max_vel = 10.0
        max_gain = np.max(np.abs(self.channel_gains)) + epsilon
        
        return np.concatenate([
            self.uav_position / max_pos,
            self.prev_velocity / max_vel,
            self.channel_gains / max_gain
        ])

def run_demo_training():
    print("Running Improved UAV Training Demo")
    print("="*60)
    
    env = UAVEnvironmentDemo()
    
    # Training metrics
    all_rewards = []
    all_users_served = []
    all_power_efficiency = []
    all_trajectories = []
    
    # Run training episodes
    n_episodes = 100
    for episode in range(n_episodes):
        obs = env.reset()
        episode_rewards = []
        episode_users = []
        episode_efficiency = []
        
        for step in range(1000):
            obs, reward, done, info = env.step()
            
            episode_rewards.append(reward)
            episode_users.append(info['users_served'])
            episode_efficiency.append(info['power_efficiency'])
            
            if done:
                break
        
        env.episode_count += 1
        all_rewards.append(np.mean(episode_rewards))
        all_users_served.append(np.mean(episode_users))
        all_power_efficiency.append(np.mean(episode_efficiency))
        all_trajectories.append(np.array(env.trajectory))
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}: Reward={np.mean(episode_rewards):.1f}, "
                  f"Users={np.mean(episode_users):.1f}, "
                  f"Efficiency={np.mean(episode_efficiency):.2%}")
    
    # Create comprehensive plots
    create_training_plots(all_rewards, all_users_served, all_power_efficiency, all_trajectories)
    
    # Show final trajectory
    create_trajectory_visualization(all_trajectories[-1], env.users_positions)
    
    print("\nTraining complete! Check the generated plots.")

def create_training_plots(rewards, users_served, efficiency, trajectories):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    episodes = range(1, len(rewards) + 1)
    
    # Reward curve
    axes[0, 0].plot(episodes, rewards, 'b-', alpha=0.7)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Average Reward')
    axes[0, 0].set_title('Training Reward Curve')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Users served
    axes[0, 1].plot(episodes, users_served, 'g-', alpha=0.7)
    axes[0, 1].axhline(y=85, color='r', linestyle='--', label='Target: 85 users')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Average Users Served')
    axes[0, 1].set_title('Users Served Over Training')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Power efficiency
    axes[0, 2].plot(episodes, np.array(efficiency) * 100, 'm-', alpha=0.7)
    axes[0, 2].axhline(y=90, color='r', linestyle='--', label='Target: 90%')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Power Efficiency (%)')
    axes[0, 2].set_title('Power Efficiency Over Training')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Moving averages
    window = 10
    if len(episodes) >= window:
        moving_avg_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
        moving_avg_users = np.convolve(users_served, np.ones(window)/window, mode='valid')
        moving_avg_efficiency = np.convolve(efficiency, np.ones(window)/window, mode='valid')
        
        axes[1, 0].plot(episodes[window-1:], moving_avg_rewards, 'b-', linewidth=2)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Moving Avg Reward')
        axes[1, 0].set_title(f'{window}-Episode Moving Average Reward')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(episodes[window-1:], moving_avg_users, 'g-', linewidth=2)
        axes[1, 1].axhline(y=85, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Moving Avg Users')
        axes[1, 1].set_title(f'{window}-Episode Moving Average Users Served')
        axes[1, 1].grid(True, alpha=0.3)
        
        axes[1, 2].plot(episodes[window-1:], moving_avg_efficiency * 100, 'm-', linewidth=2)
        axes[1, 2].axhline(y=90, color='r', linestyle='--')
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('Moving Avg Efficiency (%)')
        axes[1, 2].set_title(f'{window}-Episode Moving Average Power Efficiency')
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('improved_training_results.png', dpi=150)
    plt.close()

def create_trajectory_visualization(trajectory, user_positions):
    fig = plt.figure(figsize=(20, 15))
    
    # 3D trajectory
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'b-', alpha=0.8, linewidth=2)
    ax1.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], 
               color='green', s=200, marker='o', label='Start', edgecolors='black', linewidth=2)
    ax1.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], 
               color='red', s=200, marker='*', label='End', edgecolors='black', linewidth=2)
    
    # Plot users
    ax1.scatter(user_positions[:, 0], user_positions[:, 1], user_positions[:, 2],
               color='gray', s=50, alpha=0.6, label='Users')
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('UAV Trajectory - 3D View')
    ax1.legend()
    ax1.set_zlim([0, 1100])
    
    # Top view with density
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.hexbin(trajectory[:, 0], trajectory[:, 1], gridsize=20, cmap='Blues', alpha=0.6)
    ax2.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=0.5, linewidth=1)
    ax2.scatter(user_positions[:, 0], user_positions[:, 1], 
               color='gray', s=20, alpha=0.3)
    ax2.scatter(trajectory[0, 0], trajectory[0, 1], 
               color='green', s=150, marker='o', edgecolors='black', linewidth=2)
    ax2.scatter(trajectory[-1, 0], trajectory[-1, 1], 
               color='red', s=150, marker='*', edgecolors='black', linewidth=2)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('UAV Coverage Density')
    ax2.set_xlim([0, 1000])
    ax2.set_ylim([0, 1000])
    ax2.grid(True, alpha=0.3)
    
    # Altitude profile
    ax3 = fig.add_subplot(2, 3, 3)
    timesteps = range(len(trajectory))
    ax3.plot(timesteps, trajectory[:, 2], 'purple', alpha=0.8, linewidth=2)
    ax3.axhline(y=1000, color='red', linestyle='--', linewidth=2, label='Max Alt (1km)')
    ax3.axhline(y=10, color='orange', linestyle='--', linewidth=2, label='Min Alt')
    ax3.fill_between(timesteps, 10, 1000, alpha=0.1, color='green', label='Safe Zone')
    ax3.set_xlabel('Timestep')
    ax3.set_ylabel('Altitude (m)')
    ax3.set_title('Altitude Profile')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_ylim([0, 1100])
    
    # Velocity profile
    ax4 = fig.add_subplot(2, 3, 4)
    velocities = np.linalg.norm(np.diff(trajectory, axis=0) / 0.1, axis=1)
    ax4.plot(range(len(velocities)), velocities, 'blue', alpha=0.8, linewidth=2)
    ax4.axhline(y=3.0, color='red', linestyle='--', linewidth=2, label='Max Velocity (3 m/s)')
    ax4.fill_between(range(len(velocities)), 0, 3.0, alpha=0.1, color='green', label='Safe Zone')
    ax4.set_xlabel('Timestep')
    ax4.set_ylabel('Velocity (m/s)')
    ax4.set_title('Velocity Profile')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_ylim([0, 3.5])
    
    # Movement heatmap
    ax5 = fig.add_subplot(2, 3, 5)
    H, xedges, yedges = np.histogram2d(trajectory[:, 0], trajectory[:, 1], bins=30)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax5.imshow(H.T, origin='lower', extent=extent, cmap='hot', interpolation='gaussian')
    ax5.set_xlabel('X (m)')
    ax5.set_ylabel('Y (m)')
    ax5.set_title('UAV Position Heatmap')
    plt.colorbar(im, ax=ax5, label='Time spent')
    
    # Path segments colored by time
    ax6 = fig.add_subplot(2, 3, 6)
    for i in range(len(trajectory) - 1):
        color = plt.cm.viridis(i / len(trajectory))
        ax6.plot(trajectory[i:i+2, 0], trajectory[i:i+2, 1], color=color, linewidth=2)
    
    # Add colorbar for time
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=len(trajectory)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax6, label='Timestep')
    
    ax6.scatter(user_positions[:, 0], user_positions[:, 1], 
               color='gray', s=20, alpha=0.3, label='Users')
    ax6.scatter(trajectory[0, 0], trajectory[0, 1], 
               color='green', s=150, marker='o', edgecolors='black', linewidth=2, label='Start')
    ax6.scatter(trajectory[-1, 0], trajectory[-1, 1], 
               color='red', s=150, marker='*', edgecolors='black', linewidth=2, label='End')
    ax6.set_xlabel('X (m)')
    ax6.set_ylabel('Y (m)')
    ax6.set_title('UAV Path Colored by Time')
    ax6.set_xlim([0, 1000])
    ax6.set_ylim([0, 1000])
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    plt.tight_layout()
    plt.savefig('final_trajectory_visualization_improved.png', dpi=200)
    plt.close()

if __name__ == "__main__":
    run_demo_training()