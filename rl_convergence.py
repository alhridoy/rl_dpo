import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import time
from collections import deque


class UAVEnvironment(gym.Env):
    
    def __init__(self):
        super(UAVEnvironment, self).__init__()
        self.num_users = 100  
        self.num_subchannels = 5  
        self.W_s = 100  
        self.N_0 = (10 ** (-80 / 10)) / 1000  
        self.max_power = 1.0  
        self.R_min = 3.0  
        self.time_granularity = 0.1  
        self.max_acceleration = 10  
        self.z_movement_factor = 2  

        self.uav_position = np.array([500, 500, 100])  
        self.prev_velocity = np.array([0.0, 0.0, 0.0])  
        
        # Convergence parameters
        self.power_waste_penalty_weight = 0.5  # Weight for power waste penalty
        self.convergence_window = 50  # Episodes to check for convergence
        self.convergence_threshold = 1  # Allowed variance in users served
        self.min_power_efficiency = 0.95  # 95% power efficiency for convergence
        
        self.action_space = Box(
            low=np.zeros(6 + 2 * self.num_users * self.num_subchannels),  
            high=np.ones(6 + 2 * self.num_users * self.num_subchannels),  
            dtype=np.float32
        )

        obs_size = 3 + 3 + self.num_users  
        self.observation_space = Box(low=0, high=1, shape=(obs_size,), dtype=np.float32)

        # Tracking metrics
        self.position_violations = 0
        self.acceleration_violations = 0
        self.users_served_per_timestep = []
        self.throughput_tracker = np.zeros((self.num_users,))
        
        # Convergence tracking
        self.users_served_history = deque(maxlen=self.convergence_window)
        self.power_efficiency_history = deque(maxlen=self.convergence_window)
        self.power_waste_per_episode = []
        self.convergence_achieved = False
        self.convergence_episode = None
        
        self.reset(seed=42)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        reset_position = True
        if options is not None and 'reset_position' in options:
            reset_position = options['reset_position']

        if reset_position:
            self.uav_position = np.array([500.0, 500.0, 100.0])
            self.prev_velocity = np.array([0.0, 0.0, 0.0])

        if reset_position or not hasattr(self, 'users_positions'):
            self.users_positions = np.random.uniform(0, 1000, size=(self.num_users, 2))
            self.users_positions = np.column_stack((self.users_positions, np.zeros(self.num_users)))

        self._update_channel_gains()

        self.step_count = 0
        self.cumulative_reward = 0
        self.episode_power_waste = 0
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

        # Sample and scale acceleration
        a_sample = np.random.beta(a_mean + 1, a_var + 1)
        theta_sample = np.random.beta(theta_mean + 1, theta_var + 1)
        phi_sample = np.random.beta(phi_mean + 1, phi_var + 1)

        a = np.log(1 + np.exp(a_sample)) * self.max_acceleration  
        theta = (np.log(1 + np.exp(theta_sample)) * 2 * np.pi) - np.pi  
        phi = (np.log(1 + np.exp(phi_sample)) * 2 * np.pi) - np.pi  

        ax = a * np.sin(theta) * np.cos(phi)
        ay = a * np.sin(theta) * np.sin(phi)
        az = a * np.cos(theta) * self.z_movement_factor  

        acceleration = np.array([ax, ay, az])

        # Update velocity and position
        self.prev_velocity += self.time_granularity * acceleration
        self.uav_position += self.prev_velocity * self.time_granularity

        # Clip position
        self.uav_position[2] = max(self.uav_position[2], 10)  
        
        # Process power allocation
        power_means = action[6: 6 + self.num_users * self.num_subchannels]
        power_vars = action[6 + self.num_users * self.num_subchannels: ]  

        power_means = power_means.reshape(self.num_users, self.num_subchannels)
        power_vars = power_vars.reshape(self.num_users, self.num_subchannels)

        self.power_allocations = self._apply_softmax_power_allocation(np.stack([power_means, power_vars], axis=-1))

        # Calculate throughputs and identify served/unserved users
        throughputs = self._calculate_throughputs()
        served_mask = throughputs >= self.R_min
        users_served = np.sum(served_mask)
        
        # Calculate power waste penalty
        total_power_to_served = np.sum(self.power_allocations[served_mask])
        total_power_to_unserved = np.sum(self.power_allocations[~served_mask])
        total_power_allocated = np.sum(self.power_allocations)
        
        power_efficiency = total_power_to_served / (total_power_allocated + 1e-8)
        power_waste_penalty = total_power_to_unserved / (self.max_power * self.num_users)
        
        # Modified reward function
        reward = users_served - self.power_waste_penalty_weight * power_waste_penalty
        
        # Track metrics
        self.users_served_per_timestep.append(users_served)
        self.throughput_tracker += throughputs
        self.episode_power_waste += power_waste_penalty
        self.episode_power_efficiency_sum += power_efficiency

        self.cumulative_reward += reward
        self.step_count += 1
        terminated = self.step_count >= 10000  # 10,000 timesteps per episode
        truncated = False
        
        # Episode end tracking
        if terminated:
            avg_power_efficiency = self.episode_power_efficiency_sum / self.step_count
            self.users_served_history.append(users_served)
            self.power_efficiency_history.append(avg_power_efficiency)
            self.power_waste_per_episode.append(self.episode_power_waste / self.step_count)
            
            # Check convergence
            if len(self.users_served_history) >= self.convergence_window:
                users_served_variance = np.var(self.users_served_history)
                avg_power_efficiency = np.mean(self.power_efficiency_history)
                
                if (users_served_variance <= self.convergence_threshold and 
                    avg_power_efficiency >= self.min_power_efficiency and
                    not self.convergence_achieved):
                    self.convergence_achieved = True
                    self.convergence_episode = len(self.power_waste_per_episode)

        info = {
            'users_served': users_served,
            'power_efficiency': power_efficiency,
            'power_waste': power_waste_penalty,
            'throughputs': throughputs
        }

        return self._get_state(), reward, terminated, truncated, info

    def _calculate_throughputs(self):
        throughputs = np.zeros(self.num_users)
        for user in range(self.num_users):
            total_power = np.sum(self.power_allocations[user]) if hasattr(self, 'power_allocations') else self.max_power
            snr = (self.channel_gains[user] * total_power) / self.N_0
            throughputs[user] = self.W_s * np.log2(1 + snr)
        return throughputs
    
    def _apply_softmax_power_allocation(self, power_params):
        power_means = power_params[:, :, 0]  
        power_vars = power_params[:, :, 1]  

        power_samples = np.random.beta(power_means + 1, power_vars + 1)
        power_activated = np.log(1 + np.exp(power_samples))

        exp_powers = np.exp(np.clip(power_activated - np.max(power_activated, axis=1, keepdims=True), -20, 20))
        return (exp_powers / np.sum(exp_powers, axis=1, keepdims=True)) * self.max_power

    def _get_state(self):
        epsilon = 1e-8
        max_pos = np.max(np.abs(self.uav_position)) + epsilon
        max_vel = np.max(np.abs(self.prev_velocity)) + epsilon
        max_gain = np.max(np.abs(self.channel_gains)) + epsilon

        return np.concatenate([
            self.uav_position / max_pos,
            self.prev_velocity / max_vel,
            self.channel_gains / max_gain
        ]).astype(np.float32)
        
    def print_power_allocations(self):
        print("\nPower Allocations Analysis:")
        if hasattr(self, 'power_allocations'):
            throughputs = self._calculate_throughputs()
            served_mask = throughputs >= self.R_min
            
            print(f"Users served: {np.sum(served_mask)}/{self.num_users}")
            
            # Show power allocation for served vs unserved users
            served_power = self.power_allocations[served_mask]
            unserved_power = self.power_allocations[~served_mask]
            
            if len(served_power) > 0:
                avg_power_served = np.mean(np.sum(served_power, axis=1))
                print(f"Average power to served users: {avg_power_served:.4f}")
            
            if len(unserved_power) > 0:
                avg_power_unserved = np.mean(np.sum(unserved_power, axis=1))
                print(f"Average power to unserved users: {avg_power_unserved:.4f}")
                
            # Power efficiency
            total_power_to_served = np.sum(served_power)
            total_power_allocated = np.sum(self.power_allocations)
            power_efficiency = total_power_to_served / (total_power_allocated + 1e-8)
            print(f"Power efficiency: {power_efficiency:.2%}")
            
            # Show first 5 served and unserved users
            served_indices = np.where(served_mask)[0][:5]
            unserved_indices = np.where(~served_mask)[0][:5]
            
            print("\nSample served users:")
            for idx in served_indices:
                print(f"  User {idx}: Power={np.sum(self.power_allocations[idx]):.4f}, Throughput={throughputs[idx]:.2f}")
                
            if len(unserved_indices) > 0:
                print("\nSample unserved users:")
                for idx in unserved_indices:
                    print(f"  User {idx}: Power={np.sum(self.power_allocations[idx]):.4f}, Throughput={throughputs[idx]:.2f}")
        else:
            print("Power allocations not yet initialized.")

    def print_summary(self, episode):
        print(f"\n{'='*60}")
        print(f"Episode {episode} Summary")
        print(f"{'='*60}")
        
        if len(self.users_served_per_timestep) > 0:
            avg_users_served = np.mean(self.users_served_per_timestep)
            print(f"Average users served per timestep: {avg_users_served:.2f}")
            
            # Convergence metrics
            if len(self.users_served_history) > 0:
                recent_users_served = list(self.users_served_history)[-10:]
                print(f"Recent users served (last 10 episodes): {recent_users_served}")
                print(f"Variance in users served: {np.var(recent_users_served):.2f}")
            
            if len(self.power_efficiency_history) > 0:
                recent_efficiency = list(self.power_efficiency_history)[-10:]
                print(f"Average power efficiency (last 10 episodes): {np.mean(recent_efficiency):.2%}")
            
            if self.convergence_achieved:
                print(f"\n*** CONVERGENCE ACHIEVED at episode {self.convergence_episode} ***")
            
            # Reset per-episode metrics
            self.users_served_per_timestep = []
            self.throughput_tracker = np.zeros((self.num_users,))


# Create environment
env = Monitor(UAVEnvironment())

# Define policy network architecture
policy_kwargs = dict(
    net_arch=[512, 512, 256],
    activation_fn=torch.nn.ReLU,
)

model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,
    learning_rate=0.0001,
    gamma=0.99,
    gae_lambda=0.98,
    n_steps=10000,
    batch_size=1000,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=1,
    tensorboard_log=f"./ppo_uav_convergence_tensorboard/PPO_{int(time.time())}"
)

# Training parameters
total_timesteps = 1000000
timesteps_per_episode = 10000  # 10,000 timesteps per episode
total_episodes = total_timesteps // timesteps_per_episode  # 100 episodes
episodes_per_checkpoint = 10  # Print every 10 episodes
user_distribution_reset_interval = 200  # Reset every 200 episodes (but we only have 100)

# Initialize tracking
current_episode = 0
all_users_served = []
all_power_efficiency = []
all_power_waste = []

# Training loop
print(f"Starting training for {total_timesteps} timesteps ({total_episodes} episodes)")
print(f"User distribution will be reset every {user_distribution_reset_interval} episodes")
print(f"Target convergence: Stable 85+ users served with >95% power efficiency\n")

while current_episode < total_episodes:
    episodes_in_chunk = min(episodes_per_checkpoint, total_episodes - current_episode)
    timesteps_in_chunk = episodes_in_chunk * timesteps_per_episode
    
    # Check if we need to reset user distribution
    if current_episode > 0 and current_episode % user_distribution_reset_interval == 0:
        print(f"\n*** Resetting user distribution at episode {current_episode} ***\n")
        obs, _ = env.reset(options={'reset_position': True})
    else:
        obs, _ = env.reset(options={'reset_position': False})
    
    # Train the model
    model.learn(total_timesteps=timesteps_in_chunk, reset_num_timesteps=False)
    current_episode += episodes_in_chunk

    # Print detailed analysis every checkpoint
    if current_episode % episodes_per_checkpoint == 0 or current_episode >= total_episodes:
        print(f"\nCheckpoint at Episode {current_episode}")
        env.unwrapped.print_power_allocations()
        env.unwrapped.print_summary(current_episode)
        
        # Plot convergence metrics if we have enough data
        if current_episode >= 20:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Users served over time
            ax = axes[0, 0]
            episodes = list(range(1, len(env.unwrapped.power_waste_per_episode) + 1))
            users_history = list(env.unwrapped.users_served_history)
            ax.plot(episodes[-len(users_history):], users_history, 'b-', linewidth=2)
            ax.axhline(y=85, color='r', linestyle='--', label='Target: 85 users')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Users Served')
            ax.set_title('Users Served Over Time')
            ax.legend()
            ax.grid(True)
            
            # Power efficiency over time
            ax = axes[0, 1]
            efficiency_history = list(env.unwrapped.power_efficiency_history)
            ax.plot(episodes[-len(efficiency_history):], 
                   np.array(efficiency_history) * 100, 'g-', linewidth=2)
            ax.axhline(y=95, color='r', linestyle='--', label='Target: 95%')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Power Efficiency (%)')
            ax.set_title('Power Efficiency Over Time')
            ax.legend()
            ax.grid(True)
            
            # Power waste over time
            ax = axes[1, 0]
            ax.plot(episodes, env.unwrapped.power_waste_per_episode, 'm-', linewidth=2)
            ax.set_xlabel('Episode')
            ax.set_ylabel('Average Power Waste')
            ax.set_title('Power Waste Over Time')
            ax.grid(True)
            
            # Convergence indicator
            ax = axes[1, 1]
            if len(users_history) >= env.unwrapped.convergence_window:
                variance_history = []
                for i in range(env.unwrapped.convergence_window, len(users_history) + 1):
                    variance = np.var(users_history[i-env.unwrapped.convergence_window:i])
                    variance_history.append(variance)
                
                ax.plot(episodes[-len(variance_history):], variance_history, 'r-', linewidth=2)
                ax.axhline(y=env.unwrapped.convergence_threshold, color='g', 
                          linestyle='--', label=f'Threshold: {env.unwrapped.convergence_threshold}')
                ax.set_xlabel('Episode')
                ax.set_ylabel('Variance in Users Served')
                ax.set_title(f'Convergence Metric (Window: {env.unwrapped.convergence_window} episodes)')
                ax.legend()
                ax.grid(True)
            
            plt.tight_layout()
            plt.savefig(f'convergence_metrics_episode_{current_episode}.png')
            plt.show()
        
        # Simulate one episode for trajectory visualization
        obs, _ = env.reset(options={'reset_position': False})
        initial_position = env.unwrapped.uav_position.copy()
        uav_positions = [initial_position.copy()]
        
        for _ in range(1000):  # Sample first 1000 steps for visualization
            action, _ = model.predict(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            uav_positions.append(env.unwrapped.uav_position.copy())
            if terminated or truncated:
                break
        
        final_position = env.unwrapped.uav_position.copy()
        
        # Plot UAV trajectory
        uav_positions = np.array(uav_positions)
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Color trajectory by time
        colors = plt.cm.viridis(np.linspace(0, 1, len(uav_positions)))
        for i in range(len(uav_positions) - 1):
            ax.plot(uav_positions[i:i+2, 0], uav_positions[i:i+2, 1], 
                   uav_positions[i:i+2, 2], color=colors[i], alpha=0.8)
        
        # Plot users with different colors for served/unserved
        throughputs = env.unwrapped._calculate_throughputs()
        served_mask = throughputs >= env.unwrapped.R_min
        
        ax.scatter(env.unwrapped.users_positions[served_mask, 0], 
                  env.unwrapped.users_positions[served_mask, 1], 
                  env.unwrapped.users_positions[served_mask, 2], 
                  color="green", s=50, label=f"Served Users ({np.sum(served_mask)})")
        
        ax.scatter(env.unwrapped.users_positions[~served_mask, 0], 
                  env.unwrapped.users_positions[~served_mask, 1], 
                  env.unwrapped.users_positions[~served_mask, 2], 
                  color="red", s=50, label=f"Unserved Users ({np.sum(~served_mask)})")
        
        ax.scatter(initial_position[0], initial_position[1], initial_position[2], 
                  color="blue", s=200, marker='o', label="Start", edgecolors='black')
        ax.scatter(final_position[0], final_position[1], final_position[2], 
                  color="purple", s=200, marker='*', label="End", edgecolors='black')
        
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.set_zlabel("Z Position (m)")
        ax.set_title(f"UAV Trajectory - Episode {current_episode}\n" + 
                    f"Users Served: {np.sum(served_mask)}/{env.unwrapped.num_users}")
        ax.legend()
        
        # Set axis limits for better visualization
        ax.set_xlim([0, 1000])
        ax.set_ylim([0, 1000])
        ax.set_zlim([0, 200])
        
        plt.tight_layout()
        plt.savefig(f'trajectory_episode_{current_episode}.png')
        plt.show()

# Final summary
print(f"\n{'='*60}")
print("TRAINING COMPLETE")
print(f"{'='*60}")
if env.unwrapped.convergence_achieved:
    print(f"Convergence achieved at episode {env.unwrapped.convergence_episode}")
else:
    print("Convergence not achieved within training period")

# Save the model
model.save(f"ppo_uav_convergence_model_{int(time.time())}")
print(f"\nModel saved to ppo_uav_convergence_model_{int(time.time())}.zip")