import numpy as np
import matplotlib.pyplot as plt
from collections import deque

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
        
        self.reset()

    def reset(self):
        # Random starting position
        self.uav_position = np.array([
            500.0 + np.random.uniform(-100, 100),
            500.0 + np.random.uniform(-100, 100),
            100.0 + np.random.uniform(-20, 20)
        ])
        self.prev_velocity = np.array([0.0, 0.0, 0.0])
        
        # Generate clustered user distribution
        self.users_positions = self._generate_clustered_users()
        self._update_channel_gains()
        
        self.step_count = 0
        self.trajectory = [self.uav_position.copy()]
        
    def _generate_clustered_users(self):
        # Create 3 clusters
        clusters = []
        cluster_centers = np.array([[200, 200], [800, 800], [500, 800]])
        
        for center in cluster_centers:
            cluster = np.random.normal(center, 100, size=(33, 2))
            clusters.append(np.clip(cluster, 0, 1000))
        
        positions = np.vstack(clusters + [np.random.uniform(0, 1000, size=(1, 2))])
        return np.column_stack((positions, np.zeros(self.num_users)))

    def _update_channel_gains(self):
        H = self.uav_position[2]
        r_nj = np.linalg.norm(self.users_positions[:, :2] - self.uav_position[:2], axis=1)
        
        # Simplified channel model
        d_nj = np.sqrt(H**2 + r_nj**2)
        PL = 20 * np.log10(d_nj) + 50  # Simplified path loss
        self.channel_gains = 10 ** (-PL / 10)

    def simulate_episode(self):
        episode_metrics = {
            'rewards': [],
            'users_served': [],
            'power_efficiency': [],
            'trajectory': []
        }
        
        for step in range(100):  # Reduced steps for faster demo
            # Smart movement towards underserved areas
            throughputs = self._calculate_throughputs()
            served_mask = throughputs >= self.R_min
            
            if np.any(~served_mask):
                # Move towards nearest unserved user
                unserved_positions = self.users_positions[~served_mask, :2]
                distances = np.linalg.norm(unserved_positions - self.uav_position[:2], axis=1)
                target = unserved_positions[np.argmin(distances)]
                
                direction = target - self.uav_position[:2]
                direction = direction / (np.linalg.norm(direction) + 1e-8)
                
                ax = direction[0] * 0.8
                ay = direction[1] * 0.8
                az = -0.05 if self.uav_position[2] > 80 else 0.05  # Optimize altitude
            else:
                # Explore
                ax = np.random.uniform(-0.5, 0.5)
                ay = np.random.uniform(-0.5, 0.5)
                az = 0
            
            # Update physics
            acceleration = np.array([ax, ay, az])
            new_velocity = self.prev_velocity + self.time_granularity * acceleration
            
            # Apply velocity constraint
            vel_mag = np.linalg.norm(new_velocity)
            if vel_mag > self.max_velocity:
                new_velocity = new_velocity * (self.max_velocity / vel_mag)
            
            self.prev_velocity = new_velocity
            
            # Update position
            self.uav_position += self.prev_velocity * self.time_granularity
            self.uav_position[0] = np.clip(self.uav_position[0], 0, 1000)
            self.uav_position[1] = np.clip(self.uav_position[1], 0, 1000)
            self.uav_position[2] = np.clip(self.uav_position[2], self.min_altitude, self.max_altitude)
            
            self.trajectory.append(self.uav_position.copy())
            self._update_channel_gains()
            
            # Smart power allocation
            self.power_allocations = self._allocate_power()
            
            # Calculate metrics
            throughputs = self._calculate_throughputs()
            served_mask = throughputs >= self.R_min
            users_served = np.sum(served_mask)
            
            power_to_served = np.sum(self.power_allocations[served_mask])
            total_power = np.sum(self.power_allocations)
            power_efficiency = power_to_served / (total_power + 1e-8)
            
            # Simple reward
            reward = users_served * 0.5 - (1 - power_efficiency) * 50
            
            episode_metrics['rewards'].append(reward)
            episode_metrics['users_served'].append(users_served)
            episode_metrics['power_efficiency'].append(power_efficiency)
        
        episode_metrics['trajectory'] = np.array(self.trajectory)
        return episode_metrics

    def _calculate_throughputs(self):
        throughputs = np.zeros(self.num_users)
        for user in range(self.num_users):
            power = self.power_allocations[user].sum() if hasattr(self, 'power_allocations') else self.max_power
            snr = (self.channel_gains[user] * power) / self.N_0
            throughputs[user] = self.W_s * np.log2(1 + snr)
        return throughputs
    
    def _allocate_power(self):
        # Smart allocation based on channel quality
        min_power_needed = (self.N_0 * (2**(self.R_min/self.W_s) - 1)) / (self.channel_gains + 1e-8)
        servable = min_power_needed < self.max_power * 0.7
        
        allocations = np.zeros((self.num_users, self.num_subchannels))
        
        if np.any(servable):
            # Allocate to servable users
            servable_idx = np.where(servable)[0]
            power_per_user = (self.max_power * self.num_users * 0.9) / len(servable_idx)
            
            for idx in servable_idx:
                allocations[idx] = np.ones(self.num_subchannels) * power_per_user / self.num_subchannels
        
        # Minimal power to others
        allocations[~servable] = 0.001 / self.num_subchannels
        
        return allocations

def main():
    print("UAV Training Demo - Improved Version")
    print("="*60)
    
    # Simulate training
    n_episodes = 20
    all_metrics = []
    
    for episode in range(n_episodes):
        env = UAVEnvironmentDemo()
        metrics = env.simulate_episode()
        all_metrics.append(metrics)
        
        avg_reward = np.mean(metrics['rewards'])
        avg_users = np.mean(metrics['users_served'])
        avg_efficiency = np.mean(metrics['power_efficiency'])
        
        print(f"Episode {episode + 1}: Reward={avg_reward:.1f}, "
              f"Users={avg_users:.1f}/{env.num_users}, "
              f"Efficiency={avg_efficiency:.2%}")
    
    # Create plots
    create_plots(all_metrics)
    
    print("\nDemo complete! Check the generated plots:")
    print("- training_curves_demo.png")
    print("- final_trajectory_demo.png")

def create_plots(all_metrics):
    # Training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    episodes = range(1, len(all_metrics) + 1)
    avg_rewards = [np.mean(m['rewards']) for m in all_metrics]
    avg_users = [np.mean(m['users_served']) for m in all_metrics]
    avg_efficiency = [np.mean(m['power_efficiency']) for m in all_metrics]
    
    # Reward curve
    axes[0].plot(episodes, avg_rewards, 'b-o', markersize=4)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Average Reward')
    axes[0].set_title('Training Rewards')
    axes[0].grid(True, alpha=0.3)
    
    # Users served
    axes[1].plot(episodes, avg_users, 'g-o', markersize=4)
    axes[1].axhline(y=85, color='r', linestyle='--', label='Target')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Users Served')
    axes[1].set_title('Average Users Served')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 100])
    
    # Power efficiency
    axes[2].plot(episodes, np.array(avg_efficiency) * 100, 'm-o', markersize=4)
    axes[2].axhline(y=90, color='r', linestyle='--', label='Target')
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Efficiency (%)')
    axes[2].set_title('Power Efficiency')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig('training_curves_demo.png', dpi=150)
    plt.close()
    
    # Final trajectory visualization
    last_trajectory = all_metrics[-1]['trajectory']
    last_env = UAVEnvironmentDemo()  # To get user positions
    
    fig = plt.figure(figsize=(15, 10))
    
    # 3D trajectory
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(last_trajectory[:, 0], last_trajectory[:, 1], last_trajectory[:, 2], 
             'b-', alpha=0.8, linewidth=2)
    ax1.scatter(last_trajectory[0, 0], last_trajectory[0, 1], last_trajectory[0, 2], 
               color='green', s=200, marker='o', label='Start', edgecolors='black')
    ax1.scatter(last_trajectory[-1, 0], last_trajectory[-1, 1], last_trajectory[-1, 2], 
               color='red', s=200, marker='*', label='End', edgecolors='black')
    
    # Add users
    ax1.scatter(last_env.users_positions[:, 0], 
               last_env.users_positions[:, 1], 
               last_env.users_positions[:, 2],
               color='gray', s=30, alpha=0.5, label='Users')
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)') 
    ax1.set_zlabel('Z (m)')
    ax1.set_title('UAV Trajectory - Episode 20')
    ax1.legend()
    
    # Top view
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(last_trajectory[:, 0], last_trajectory[:, 1], 'b-', alpha=0.8)
    ax2.scatter(last_env.users_positions[:, 0], 
               last_env.users_positions[:, 1],
               color='gray', s=30, alpha=0.5)
    ax2.scatter(last_trajectory[0, 0], last_trajectory[0, 1], 
               color='green', s=150, marker='o', edgecolors='black')
    ax2.scatter(last_trajectory[-1, 0], last_trajectory[-1, 1], 
               color='red', s=150, marker='*', edgecolors='black')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Top View')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1000])
    ax2.set_ylim([0, 1000])
    
    # Altitude profile
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(range(len(last_trajectory)), last_trajectory[:, 2], 'purple', linewidth=2)
    ax3.axhline(y=1000, color='red', linestyle='--', label='Max Alt')
    ax3.axhline(y=10, color='orange', linestyle='--', label='Min Alt')
    ax3.set_xlabel('Timestep')
    ax3.set_ylabel('Altitude (m)')
    ax3.set_title('Altitude Profile')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Performance over episode
    ax4 = fig.add_subplot(2, 2, 4)
    timesteps = range(len(all_metrics[-1]['users_served']))
    ax4.plot(timesteps, all_metrics[-1]['users_served'], 'g-', label='Users Served')
    ax4.plot(timesteps, np.array(all_metrics[-1]['power_efficiency']) * 100, 
             'm-', label='Power Efficiency %')
    ax4.set_xlabel('Timestep')
    ax4.set_ylabel('Value')
    ax4.set_title('Performance During Episode')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('final_trajectory_demo.png', dpi=150)
    plt.close()

if __name__ == "__main__":
    main()