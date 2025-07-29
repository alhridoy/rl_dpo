import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import time


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
        self.max_acceleration = 10  # INCREASED for more UAV movement  
        self.z_movement_factor = 2  # New: Ensures more vertical movement

        self.uav_position = np.array([500, 500, 100])  # Start at a higher altitude  
        self.prev_velocity = np.array([0.0, 0.0, 0.0])  
        
        self.action_space = Box(
            low=np.zeros(6 + 2 * self.num_users * self.num_subchannels),  
            high=np.ones(6 + 2 * self.num_users * self.num_subchannels),  
            dtype=np.float32
        )

        obs_size = 3 + 3 + self.num_users  
        self.observation_space = Box(low=0, high=1, shape=(obs_size,), dtype=np.float32)

        self.position_violations = 0
        self.acceleration_violations = 0
        self.users_served_per_timestep = []
        self.throughput_tracker = np.zeros((self.num_users,))

        self.reset(seed=42)

    def reset(self, seed=None, options=None):
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)

        # Handle reset options
        reset_position = True
        if options is not None and 'reset_position' in options:
            reset_position = options['reset_position']

        # Reset UAV position if requested
        if reset_position:
            self.uav_position = np.array([500.0, 500.0, 100.0])
            self.prev_velocity = np.array([0.0, 0.0, 0.0])

        # Always generate new user positions (or keep existing based on reset_position)
        if reset_position or not hasattr(self, 'users_positions'):
            self.users_positions = np.random.uniform(0, 1000, size=(self.num_users, 2))
            self.users_positions = np.column_stack((self.users_positions, np.zeros(self.num_users)))

        self._update_channel_gains()

        self.step_count = 0
        self.cumulative_reward = 0
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
        az = a * np.cos(theta) * self.z_movement_factor  # Z-axis scaling

        acceleration = np.array([ax, ay, az])

        # Update velocity and position
        self.prev_velocity += self.time_granularity * acceleration
        self.uav_position += self.prev_velocity * self.time_granularity

        # Clip position
        self.uav_position[2] = max(self.uav_position[2], 10)  # Ensure Z > 10
        # Process power allocation (fixed slicing)
        power_means = action[6: 6 + self.num_users * self.num_subchannels]
        power_vars = action[6 + self.num_users * self.num_subchannels: ]  # Extract remaining values

        power_means = power_means.reshape(self.num_users, self.num_subchannels)
        power_vars = power_vars.reshape(self.num_users, self.num_subchannels)

        self.power_allocations = self._apply_softmax_power_allocation(np.stack([power_means, power_vars], axis=-1))

        # Compute throughput and reward
        throughputs = self._calculate_throughputs()
        users_served = np.sum((throughputs >= self.R_min).astype(int))
        reward = users_served

        # Track statistics
        self.users_served_per_timestep.append(users_served)
        self.throughput_tracker += throughputs

        self.cumulative_reward += reward
        self.step_count += 1
        terminated = self.step_count >= 1000
        truncated = False

        return self._get_state(), reward, terminated, truncated, {}

    def _calculate_throughputs(self):
        throughputs = np.zeros(self.num_users)
        for user in range(self.num_users):
            # Calculate total power allocated to this user across all subchannels
            total_power = np.sum(self.power_allocations[user]) if hasattr(self, 'power_allocations') else self.max_power
            # Calculate SNR with power allocation
            snr = (self.channel_gains[user] * total_power) / self.N_0
            throughputs[user] = self.W_s * np.log2(1 + snr)
        return throughputs
    
    def _apply_softmax_power_allocation(self, power_params):
        power_means = power_params[:, :, 0]  
        power_vars = power_params[:, :, 1]  

        # Sample from Beta distribution
        power_samples = np.random.beta(power_means + 1, power_vars + 1)

        # Apply SoftPlus activation
        power_activated = np.log(1 + np.exp(power_samples))

        # Normalize with Softmax **per user** (avoid overflow)
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
        print("Power Allocations per User (across subchannels):")
        if hasattr(self, 'power_allocations'):
            for user in range(min(10, self.num_users)):  # Print only first 10 users to avoid spam
                user_power = self.power_allocations[user]
                user_power_sum = np.sum(user_power)
                print(f"User {user}: {user_power} (Sum: {user_power_sum:.4f})")
            if self.num_users > 10:
                print(f"... (showing first 10 out of {self.num_users} users)")
        else:
            print("Power allocations not yet initialized.")

    def print_summary(self, episode_range):
        print(f"From Episode {episode_range[0]} to {episode_range[1]} total ({episode_range[1] - episode_range[0] + 1} * 1000) timesteps,")
        print(f"Position constraint violated: {self.position_violations} times")
        print(f"Acceleration constraint violated: {self.acceleration_violations} times")

        if len(self.users_served_per_timestep) > 0:
            avg_users_served = np.mean(self.users_served_per_timestep)
            print(f"Average number of users served per timestep: {avg_users_served:.2f}")

            avg_throughputs = self.throughput_tracker / (len(self.users_served_per_timestep) + 1e-8)
            print("Average throughput per user (first 10):")
            for user in range(min(10, self.num_users)):
                print(f"User {user}: {avg_throughputs[user]:.4f}")
            if self.num_users > 10:
                print(f"... (showing first 10 out of {self.num_users} users)")
        else:
            print("No timestep data available yet.")
        print("\n")

        self.position_violations = 0
        self.acceleration_violations = 0
        self.users_served_per_timestep = []
        self.throughput_tracker = np.zeros((self.num_users,))


# Create environment
env = Monitor(UAVEnvironment())

# Define policy network architecture
policy_kwargs = dict(
    net_arch=[512, 512, 256],  # Hidden layers for both policy and value networks
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
    ent_coef=0.01,  # Entropy coefficient for exploration
    verbose=1,
    tensorboard_log=f"./ppo_uav_tensorboard/PPO_{int(time.time())}"  # Unique log for each run
)
# Train for 10,000,000 timesteps
# Train for 10,000,000 timesteps
total_timesteps = 1000000
timesteps_per_episode = 1000
total_episodes = total_timesteps // timesteps_per_episode  # 1000 episodes
episodes_per_checkpoint = 1  # Print and plot every 20 episodes

# Initialize episode counter
current_episode = 0

# Track cumulative rewards for early stopping
cumulative_rewards = []
early_stop_threshold = 0.1  # Threshold for cumulative reward change
early_stop_count = 0  # Counter for episodes with small cumulative reward change

# Training loop
while current_episode < total_episodes:
    episodes_in_chunk = min(episodes_per_checkpoint, total_episodes - current_episode)
    timesteps_in_chunk = episodes_in_chunk * timesteps_per_episode
    
    # Reset the environment only if starting a new episode
    if current_episode == 0 or current_episode % episodes_per_checkpoint == 0:
        obs, _ = env.reset(options={'reset_position': True})  # Reset UAV position
    else:
        obs, _ = env.reset(options={'reset_position': False})  # Continue from last position
    
    # Train the model
    model.learn(total_timesteps=timesteps_in_chunk, reset_num_timesteps=False)
    current_episode += episodes_in_chunk

    # Track cumulative reward
    cumulative_rewards.append(env.unwrapped.cumulative_reward)

    # Check for early stopping condition
    if len(cumulative_rewards) >= 10:
        reward_change = np.abs(cumulative_rewards[-1] - cumulative_rewards[-10])
        if reward_change < early_stop_threshold:
            early_stop_count += 1
        else:
            early_stop_count = 0

        # Change user distribution if cumulative reward change is small for 10 episodes
        if early_stop_count >= 10:
            print(f"Episode {current_episode}: User distribution has changed.")
            obs, _ = env.reset(options={'reset_position': True})  # Reset UAV position and user distribution
            early_stop_count = 0  # Reset the counter

    # Print power allocations and plot UAV movement every 20 episodes
    if current_episode % episodes_per_checkpoint == 0 or current_episode >= total_episodes:
        print(f"Episode {current_episode}: Power Allocations")
        env.unwrapped.print_power_allocations()
        
        # Print summary of constraint violations and users served
        episode_range = (current_episode - episodes_per_checkpoint + 1, current_episode)
        env.unwrapped.print_summary(episode_range)
        
        # Simulate one episode to collect UAV positions
        obs, _ = env.reset(options={'reset_position': False})  # Continue from last position
        initial_position = env.unwrapped.uav_position.copy()
        uav_positions = [initial_position.copy()]
        for _ in range(timesteps_per_episode):  # Simulate one episode
            action, _ = model.predict(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            uav_positions.append(env.unwrapped.uav_position.copy())
            if terminated or truncated:
                break
        final_position = env.unwrapped.uav_position.copy()
        
        # Plot UAV trajectory
        uav_positions = np.array(uav_positions)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(uav_positions[:, 0], uav_positions[:, 1], uav_positions[:, 2], label="UAV Trajectory", color="blue")
        ax.scatter(env.unwrapped.users_positions[:, 0], env.unwrapped.users_positions[:, 1], env.unwrapped.users_positions[:, 2], color="red", label="Users")
        ax.scatter(initial_position[0], initial_position[1], initial_position[2], color="green", s=100, label="Initial Position")
        ax.text(initial_position[0], initial_position[1], initial_position[2], "Initial", color="green")
        ax.scatter(final_position[0], final_position[1], final_position[2], color="purple", s=100, label="Final Position")
        ax.text(final_position[0], final_position[1], final_position[2], "Final", color="purple")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_zlabel("Z Position (Altitude)")
        ax.set_title(f"3D UAV Trajectory (Episode {current_episode})")
        ax.legend()
        plt.show()