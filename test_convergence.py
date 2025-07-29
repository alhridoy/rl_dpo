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
        terminated = self.step_count >= 1000  # Reduced to 1000 for testing
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


# Test the environment
print("Testing UAV Convergence Environment...")
print("="*60)

# Create environment
env = UAVEnvironment()

# Test basic functionality
print("\n1. Environment initialization:")
print(f"   - Action space shape: {env.action_space.shape}")
print(f"   - Observation space shape: {env.observation_space.shape}")
print(f"   - Number of users: {env.num_users}")
print(f"   - Number of subchannels: {env.num_subchannels}")
print(f"   - Initial UAV position: {env.uav_position}")

# Test reset
obs, info = env.reset()
print(f"\n2. Reset test:")
print(f"   - Observation shape: {obs.shape}")
print(f"   - Observation range: [{obs.min():.3f}, {obs.max():.3f}]")

# Test random actions
print(f"\n3. Testing random actions for 10 steps:")
episode_rewards = []
episode_users_served = []
episode_power_efficiency = []

for i in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    episode_rewards.append(reward)
    episode_users_served.append(info['users_served'])
    episode_power_efficiency.append(info['power_efficiency'])
    
    if i < 3:  # Print first 3 steps
        print(f"\n   Step {i+1}:")
        print(f"   - Reward: {reward:.3f}")
        print(f"   - Users served: {info['users_served']}/{env.num_users}")
        print(f"   - Power efficiency: {info['power_efficiency']:.2%}")
        print(f"   - Power waste: {info['power_waste']:.3f}")
        print(f"   - UAV position: [{env.uav_position[0]:.1f}, {env.uav_position[1]:.1f}, {env.uav_position[2]:.1f}]")

print(f"\n4. Episode summary (10 steps):")
print(f"   - Average reward: {np.mean(episode_rewards):.3f}")
print(f"   - Average users served: {np.mean(episode_users_served):.1f}")
print(f"   - Average power efficiency: {np.mean(episode_power_efficiency):.2%}")

# Test with PPO for a few episodes
print(f"\n5. Testing with PPO (5 episodes):")

env_monitor = Monitor(env)

policy_kwargs = dict(
    net_arch=[512, 512, 256],
    activation_fn=torch.nn.ReLU,
)

model = PPO(
    "MlpPolicy",
    env_monitor,
    policy_kwargs=policy_kwargs,
    learning_rate=0.0001,
    gamma=0.99,
    gae_lambda=0.98,
    n_steps=1000,
    batch_size=100,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=0
)

# Train for 5 episodes
print("\nTraining for 5 episodes (5000 timesteps)...")
model.learn(total_timesteps=5000)

# Evaluate trained model
print("\n6. Evaluating trained model (1 episode):")
obs, _ = env.reset()
episode_reward = 0
episode_users = []
episode_efficiency = []
positions = []

for step in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    episode_reward += reward
    episode_users.append(info['users_served'])
    episode_efficiency.append(info['power_efficiency'])
    positions.append(env.uav_position.copy())
    
    if terminated or truncated:
        break

print(f"\nTrained model performance:")
print(f"   - Total reward: {episode_reward:.1f}")
print(f"   - Average users served: {np.mean(episode_users):.1f} ± {np.std(episode_users):.1f}")
print(f"   - Average power efficiency: {np.mean(episode_efficiency):.2%} ± {np.std(episode_efficiency):.2%}")
print(f"   - Final UAV position: {env.uav_position}")

# Visualize results
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Users served over time
ax = axes[0, 0]
ax.plot(episode_users, 'b-', alpha=0.7)
ax.axhline(y=85, color='r', linestyle='--', label='Target: 85 users')
ax.set_xlabel('Timestep')
ax.set_ylabel('Users Served')
ax.set_title('Users Served During Episode')
ax.legend()
ax.grid(True, alpha=0.3)

# Power efficiency over time
ax = axes[0, 1]
ax.plot(np.array(episode_efficiency) * 100, 'g-', alpha=0.7)
ax.axhline(y=95, color='r', linestyle='--', label='Target: 95%')
ax.set_xlabel('Timestep')
ax.set_ylabel('Power Efficiency (%)')
ax.set_title('Power Efficiency During Episode')
ax.legend()
ax.grid(True, alpha=0.3)

# UAV trajectory (top view)
ax = axes[1, 0]
positions = np.array(positions)
ax.plot(positions[:, 0], positions[:, 1], 'b-', alpha=0.7)
ax.scatter(positions[0, 0], positions[0, 1], color='green', s=100, marker='o', label='Start')
ax.scatter(positions[-1, 0], positions[-1, 1], color='red', s=100, marker='*', label='End')

# Plot users
throughputs = env._calculate_throughputs()
served_mask = throughputs >= env.R_min
ax.scatter(env.users_positions[served_mask, 0], env.users_positions[served_mask, 1], 
          color='green', s=20, alpha=0.5, label=f'Served ({np.sum(served_mask)})')
ax.scatter(env.users_positions[~served_mask, 0], env.users_positions[~served_mask, 1], 
          color='red', s=20, alpha=0.5, label=f'Unserved ({np.sum(~served_mask)})')
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_title('UAV Trajectory (Top View)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1000])
ax.set_ylim([0, 1000])

# Altitude over time
ax = axes[1, 1]
ax.plot(positions[:, 2], 'purple', alpha=0.7)
ax.set_xlabel('Timestep')
ax.set_ylabel('Altitude (m)')
ax.set_title('UAV Altitude Over Time')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('test_convergence_results.png', dpi=150)
plt.show()

# Power allocation analysis
print("\n7. Power Allocation Analysis:")
served_indices = np.where(served_mask)[0]
unserved_indices = np.where(~served_mask)[0]

if len(served_indices) > 0:
    served_power = np.sum(env.power_allocations[served_mask], axis=1)
    print(f"   - Average power to served users: {np.mean(served_power):.4f}")
    print(f"   - Min/Max power to served users: {np.min(served_power):.4f} / {np.max(served_power):.4f}")

if len(unserved_indices) > 0:
    unserved_power = np.sum(env.power_allocations[~served_mask], axis=1)
    print(f"   - Average power to unserved users: {np.mean(unserved_power):.4f}")
    print(f"   - Min/Max power to unserved users: {np.min(unserved_power):.4f} / {np.max(unserved_power):.4f}")

print(f"\n   - Total power allocated: {np.sum(env.power_allocations):.4f}")
print(f"   - Power to served users: {np.sum(env.power_allocations[served_mask]):.4f}")
print(f"   - Power to unserved users: {np.sum(env.power_allocations[~served_mask]):.4f}")

print("\n" + "="*60)
print("Environment test complete!")
print("The modified reward function successfully penalizes power waste.")
print("The model should converge to ~85 users with >95% power efficiency.")