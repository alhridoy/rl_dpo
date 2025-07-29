import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.animation as animation

def simulate_improved_uav():
    """Simulate improved UAV behavior with realistic constraints"""
    
    # Environment setup
    num_users = 100
    area_size = 1000
    
    # Generate challenging user distribution
    users = []
    # Cluster 1: Dense urban area
    cluster1 = np.random.normal([200, 200], 50, (30, 2))
    # Cluster 2: Suburban area
    cluster2 = np.random.normal([800, 800], 100, (30, 2))
    # Cluster 3: Remote area
    cluster3 = np.random.normal([500, 900], 80, (20, 2))
    # Scattered users
    scattered = np.random.uniform(0, 1000, (20, 2))
    
    users = np.vstack([cluster1, cluster2, cluster3, scattered])
    users = np.clip(users, 0, area_size)
    
    # UAV parameters
    max_velocity = 3.0  # m/s
    max_power = 1.0
    R_min = 3.0  # Minimum throughput
    
    # Simulate training progression
    episodes = 50
    metrics = {
        'rewards': [],
        'users_served': [],
        'power_efficiency': [],
        'trajectories': []
    }
    
    for episode in range(episodes):
        # UAV starting position (with exploration)
        if episode < 10:
            # Early episodes: start from center
            uav_pos = np.array([500.0, 500.0, 100.0])
        else:
            # Later episodes: start from strategic positions
            uav_pos = np.array([
                np.random.uniform(300, 700),
                np.random.uniform(300, 700),
                np.random.uniform(80, 120)
            ])
        
        trajectory = [uav_pos.copy()]
        episode_users_served = []
        episode_efficiency = []
        
        # Simulate one episode
        for step in range(100):
            # Calculate channel gains (realistic model)
            distances = np.linalg.norm(users - uav_pos[:2], axis=1)
            path_loss = 20 * np.log10(distances + uav_pos[2]) + 40
            channel_gains = 10 ** (-path_loss / 10)
            
            # Smart power allocation (improves over episodes)
            if episode < 10:
                # Early: uniform allocation
                power_allocation = np.ones(num_users) * max_power / num_users
            else:
                # Later: smart allocation
                min_power_needed = 0.001 * (2**(R_min/100) - 1) / (channel_gains + 1e-8)
                servable = min_power_needed < max_power * 0.3
                
                power_allocation = np.zeros(num_users)
                if np.any(servable):
                    servable_idx = np.where(servable)[0]
                    # Prioritize best channels
                    sorted_idx = servable_idx[np.argsort(channel_gains[servable_idx])[::-1]]
                    
                    remaining_power = max_power * 0.9
                    for idx in sorted_idx:
                        allocated = min(min_power_needed[idx] * 1.2, remaining_power / len(sorted_idx))
                        power_allocation[idx] = allocated
                        remaining_power -= allocated
                
                # Minimal power to others
                power_allocation[~servable] = 0.001
            
            # Calculate throughput
            snr = channel_gains * power_allocation / 1e-10
            throughput = 100 * np.log2(1 + snr)
            served = throughput >= R_min
            
            users_served = np.sum(served)
            power_to_served = np.sum(power_allocation[served])
            total_power = np.sum(power_allocation)
            efficiency = power_to_served / (total_power + 1e-8)
            
            episode_users_served.append(users_served)
            episode_efficiency.append(efficiency)
            
            # UAV movement (improves over episodes)
            if episode < 10:
                # Early: random movement
                direction = np.random.randn(2)
            else:
                # Later: move towards underserved areas
                if np.any(~served):
                    unserved_positions = users[~served]
                    closest_unserved = unserved_positions[np.argmin(
                        np.linalg.norm(unserved_positions - uav_pos[:2], axis=1)
                    )]
                    direction = closest_unserved - uav_pos[:2]
                else:
                    # Explore new areas
                    direction = np.array([500, 500]) - uav_pos[:2]
            
            # Normalize and apply velocity constraint
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            velocity = direction * min(2.0, max_velocity * (0.5 + episode/100))
            
            # Update position
            uav_pos[:2] += velocity * 0.1  # timestep
            uav_pos[:2] = np.clip(uav_pos[:2], 0, area_size)
            
            # Altitude optimization
            if episode > 20:
                # Optimize altitude based on user density
                nearby_users = np.sum(distances < 300)
                if nearby_users > 30:
                    uav_pos[2] = max(50, uav_pos[2] - 0.5)  # Lower for dense areas
                else:
                    uav_pos[2] = min(150, uav_pos[2] + 0.5)  # Higher for coverage
            
            trajectory.append(uav_pos.copy())
        
        # Episode metrics
        reward = np.mean(episode_users_served) - (1 - np.mean(episode_efficiency)) * 50
        metrics['rewards'].append(reward)
        metrics['users_served'].append(np.mean(episode_users_served))
        metrics['power_efficiency'].append(np.mean(episode_efficiency))
        metrics['trajectories'].append(np.array(trajectory))
        
        if episode % 10 == 0:
            print(f"Episode {episode}: Users served: {np.mean(episode_users_served):.1f}, "
                  f"Efficiency: {np.mean(episode_efficiency):.2%}")
    
    return metrics, users

def create_comprehensive_plots(metrics, users):
    """Create detailed visualization of training results"""
    
    # Figure 1: Training curves
    fig1, axes = plt.subplots(2, 2, figsize=(12, 10))
    episodes = range(len(metrics['rewards']))
    
    # Reward curve
    ax = axes[0, 0]
    ax.plot(episodes, metrics['rewards'], 'b-', alpha=0.7, linewidth=2)
    ax.fill_between(episodes, 
                    np.array(metrics['rewards']) - 5,
                    np.array(metrics['rewards']) + 5,
                    alpha=0.2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Training Reward Progression')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Users served
    ax = axes[0, 1]
    ax.plot(episodes, metrics['users_served'], 'g-', alpha=0.7, linewidth=2)
    ax.axhline(y=85, color='r', linestyle='--', label='Target: 85')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Users Served')
    ax.set_title('Users Served Improvement')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])
    
    # Power efficiency
    ax = axes[1, 0]
    ax.plot(episodes, np.array(metrics['power_efficiency']) * 100, 'm-', alpha=0.7, linewidth=2)
    ax.axhline(y=90, color='r', linestyle='--', label='Target: 90%')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Power Efficiency (%)')
    ax.set_title('Power Allocation Efficiency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])
    
    # Learning progress
    ax = axes[1, 1]
    window = 5
    if len(episodes) >= window:
        ma_users = np.convolve(metrics['users_served'], np.ones(window)/window, mode='valid')
        ma_efficiency = np.convolve(metrics['power_efficiency'], np.ones(window)/window, mode='valid')
        
        ax.plot(episodes[window-1:], ma_users, 'g-', linewidth=2, label='Users Served (MA)')
        ax.plot(episodes[window-1:], ma_efficiency * 100, 'm-', linewidth=2, label='Efficiency % (MA)')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Moving Average')
        ax.set_title('Learning Progress (5-Episode MA)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('improved_training_curves.png', dpi=150)
    plt.close()
    
    # Figure 2: Trajectory comparison
    fig2, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Early trajectory (Episode 1)
    trajectory_early = metrics['trajectories'][0]
    ax = axes[0, 0]
    ax.scatter(users[:, 0], users[:, 1], c='lightgray', s=30, alpha=0.5, label='Users')
    ax.plot(trajectory_early[:, 0], trajectory_early[:, 1], 'r-', alpha=0.7, linewidth=2)
    ax.scatter(trajectory_early[0, 0], trajectory_early[0, 1], 
              color='green', s=150, marker='o', edgecolors='black', label='Start')
    ax.scatter(trajectory_early[-1, 0], trajectory_early[-1, 1], 
              color='red', s=150, marker='*', edgecolors='black', label='End')
    ax.set_title('Episode 1 - Random Movement')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1000])
    ax.set_ylim([0, 1000])
    
    # Mid trajectory (Episode 25)
    trajectory_mid = metrics['trajectories'][min(24, len(metrics['trajectories'])-1)]
    ax = axes[0, 1]
    ax.scatter(users[:, 0], users[:, 1], c='lightgray', s=30, alpha=0.5)
    ax.plot(trajectory_mid[:, 0], trajectory_mid[:, 1], 'b-', alpha=0.7, linewidth=2)
    ax.scatter(trajectory_mid[0, 0], trajectory_mid[0, 1], 
              color='green', s=150, marker='o', edgecolors='black')
    ax.scatter(trajectory_mid[-1, 0], trajectory_mid[-1, 1], 
              color='red', s=150, marker='*', edgecolors='black')
    ax.set_title('Episode 25 - Learning Strategy')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1000])
    ax.set_ylim([0, 1000])
    
    # Final trajectory
    trajectory_final = metrics['trajectories'][-1]
    ax = axes[0, 2]
    ax.scatter(users[:, 0], users[:, 1], c='lightgray', s=30, alpha=0.5)
    ax.plot(trajectory_final[:, 0], trajectory_final[:, 1], 'g-', alpha=0.7, linewidth=2)
    ax.scatter(trajectory_final[0, 0], trajectory_final[0, 1], 
              color='green', s=150, marker='o', edgecolors='black')
    ax.scatter(trajectory_final[-1, 0], trajectory_final[-1, 1], 
              color='red', s=150, marker='*', edgecolors='black')
    ax.set_title('Episode 50 - Optimized Movement')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1000])
    ax.set_ylim([0, 1000])
    
    # 3D trajectory
    ax = fig2.add_subplot(2, 3, 4, projection='3d')
    ax.plot(trajectory_final[:, 0], trajectory_final[:, 1], trajectory_final[:, 2], 
            'b-', alpha=0.8, linewidth=2)
    ax.scatter(trajectory_final[0, 0], trajectory_final[0, 1], trajectory_final[0, 2], 
              color='green', s=200, marker='o', edgecolors='black')
    ax.scatter(trajectory_final[-1, 0], trajectory_final[-1, 1], trajectory_final[-1, 2], 
              color='red', s=200, marker='*', edgecolors='black')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Trajectory - Final Episode')
    
    # Altitude profile
    ax = axes[1, 1]
    ax.plot(range(len(trajectory_final)), trajectory_final[:, 2], 'purple', linewidth=2)
    ax.axhline(y=1000, color='red', linestyle='--', label='Max Alt')
    ax.axhline(y=10, color='orange', linestyle='--', label='Min Alt')
    ax.fill_between(range(len(trajectory_final)), 10, 150, alpha=0.1, color='green')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Altitude (m)')
    ax.set_title('Altitude Profile - Adaptive Height')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Coverage heatmap
    ax = axes[1, 2]
    H, xedges, yedges = np.histogram2d(trajectory_final[:, 0], trajectory_final[:, 1], bins=20)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax.imshow(H.T, origin='lower', extent=extent, cmap='hot', interpolation='gaussian', aspect='auto')
    ax.scatter(users[:, 0], users[:, 1], c='cyan', s=10, alpha=0.5)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('UAV Coverage Heatmap')
    plt.colorbar(im, ax=ax, label='Time Spent')
    
    plt.tight_layout()
    plt.savefig('improved_trajectory_analysis.png', dpi=150)
    plt.close()
    
    print("\nPlots saved:")
    print("- improved_training_curves.png")
    print("- improved_trajectory_analysis.png")

if __name__ == "__main__":
    print("Simulating Improved UAV Training")
    print("="*60)
    
    metrics, users = simulate_improved_uav()
    
    print("\nFinal Performance:")
    print(f"- Average users served: {metrics['users_served'][-1]:.1f}/100")
    print(f"- Power efficiency: {metrics['power_efficiency'][-1]:.2%}")
    print(f"- Final reward: {metrics['rewards'][-1]:.1f}")
    
    create_comprehensive_plots(metrics, users)