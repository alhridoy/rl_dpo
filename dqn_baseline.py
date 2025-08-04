import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class UAVEnvironment:
    """UAV environment for resource allocation"""
    def __init__(self):
        # Environment parameters
        self.num_uavs = 3
        self.num_users = 100
        self.grid_size = 1000  # 1000x1000 meter area
        self.max_power = 30.0  # Maximum power per UAV in watts
        self.min_power = 5.0   # Minimum power per UAV
        self.max_height = 200  # Maximum UAV height in meters
        self.min_height = 50   # Minimum UAV height
        
        # Action space: [x, y, height, power] for each UAV
        self.action_dim = self.num_uavs * 4
        
        # State space: UAV positions + user densities
        self.state_dim = self.num_uavs * 3 + 25  # UAV states + 5x5 grid density
        
        # Initialize positions
        self.reset()
        
    def reset(self):
        """Reset environment to initial state"""
        # Random UAV positions
        self.uav_positions = np.random.uniform(0, self.grid_size, (self.num_uavs, 2))
        self.uav_heights = np.random.uniform(self.min_height, self.max_height, self.num_uavs)
        self.uav_powers = np.random.uniform(self.min_power, self.max_power, self.num_uavs)
        
        # Random user positions
        self.user_positions = np.random.uniform(0, self.grid_size, (self.num_users, 2))
        
        return self._get_state()
    
    def _get_state(self):
        """Get current state representation"""
        # UAV states: positions and heights
        uav_state = np.concatenate([
            self.uav_positions.flatten() / self.grid_size,  # Normalize
            self.uav_heights / self.max_height
        ])
        
        # User density in 5x5 grid
        grid_density = np.zeros((5, 5))
        cell_size = self.grid_size / 5
        
        for user_pos in self.user_positions:
            i = min(int(user_pos[0] / cell_size), 4)
            j = min(int(user_pos[1] / cell_size), 4)
            grid_density[i, j] += 1
        
        grid_density = grid_density.flatten() / self.num_users  # Normalize
        
        return np.concatenate([uav_state, grid_density])
    
    def step(self, action):
        """Execute action and return next state, reward, done"""
        # Parse action (continuous values from -1 to 1)
        action = action.reshape(self.num_uavs, 4)
        
        # Update UAV positions and power
        for i in range(self.num_uavs):
            # Position changes (limited movement)
            dx = action[i, 0] * 50  # Max 50m movement
            dy = action[i, 1] * 50
            self.uav_positions[i, 0] = np.clip(self.uav_positions[i, 0] + dx, 0, self.grid_size)
            self.uav_positions[i, 1] = np.clip(self.uav_positions[i, 1] + dy, 0, self.grid_size)
            
            # Height change
            dh = action[i, 2] * 20  # Max 20m height change
            self.uav_heights[i] = np.clip(self.uav_heights[i] + dh, self.min_height, self.max_height)
            
            # Power adjustment
            self.uav_powers[i] = self.min_power + (action[i, 3] + 1) * 0.5 * (self.max_power - self.min_power)
        
        # Calculate reward
        reward, info = self._calculate_reward()
        
        # Episode continues
        done = False
        
        return self._get_state(), reward, done, info
    
    def _calculate_reward(self):
        """Calculate reward based on coverage and power efficiency"""
        users_served = 0
        total_power_used = np.sum(self.uav_powers)
        
        # Simple coverage model
        for user_pos in self.user_positions:
            served = False
            for i in range(self.num_uavs):
                distance = np.linalg.norm(user_pos - self.uav_positions[i])
                
                # Coverage radius based on height and power
                coverage_radius = self.uav_heights[i] * 0.5 + self.uav_powers[i] * 2
                
                if distance <= coverage_radius:
                    served = True
                    break
            
            if served:
                users_served += 1
        
        # Reward components
        coverage_reward = users_served / self.num_users
        power_penalty = total_power_used / (self.num_uavs * self.max_power)
        
        # Combined reward
        reward = coverage_reward - 0.3 * power_penalty
        
        info = {
            'users_served': users_served,
            'power_efficiency': users_served / total_power_used if total_power_used > 0 else 0,
            'total_power': total_power_used
        }
        
        return reward, info


class DQN(nn.Module):
    """Simple Deep Q-Network"""
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        # Simple feed-forward network
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Output continuous actions using tanh
        return torch.tanh(self.fc3(x))


class ReplayBuffer:
    """Experience replay buffer"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state), np.array(action), np.array(reward), 
                np.array(next_state), np.array(done))
    
    def __len__(self):
        return len(self.buffer)


def train_dqn():
    """Train DQN model"""
    # Initialize environment
    env = UAVEnvironment()
    state_dim = env.state_dim
    action_dim = env.action_dim
    
    # Initialize networks
    q_network = DQN(state_dim, action_dim).to(device)
    target_network = DQN(state_dim, action_dim).to(device)
    target_network.load_state_dict(q_network.state_dict())
    
    # Optimizer
    optimizer = optim.Adam(q_network.parameters(), lr=0.001)
    
    # Replay buffer
    replay_buffer = ReplayBuffer()
    
    # Training parameters
    num_timesteps = 500000
    batch_size = 64
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995
    target_update_freq = 1000
    
    # Logging
    rewards = []
    users_served_history = []
    power_efficiency_history = []
    
    # Training loop
    state = env.reset()
    epsilon = epsilon_start
    
    for timestep in range(num_timesteps):
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = np.random.uniform(-1, 1, action_dim)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = q_network(state_tensor).cpu().numpy()[0]
        
        # Environment step
        next_state, reward, done, info = env.step(action)
        
        # Store in replay buffer
        replay_buffer.push(state, action, reward, next_state, done)
        
        # Training
        if len(replay_buffer) > batch_size:
            # Sample batch
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = replay_buffer.sample(batch_size)
            
            # Convert to tensors
            states = torch.FloatTensor(batch_state).to(device)
            actions = torch.FloatTensor(batch_action).to(device)
            rewards = torch.FloatTensor(batch_reward).to(device)
            next_states = torch.FloatTensor(batch_next_state).to(device)
            dones = torch.FloatTensor(batch_done).to(device)
            
            # Current Q values
            current_q = q_network(states)
            
            # Target Q values
            with torch.no_grad():
                next_q = target_network(next_states)
                target_q = rewards + gamma * torch.max(next_q, dim=1)[0] * (1 - dones)
            
            # Loss (simplified - treating continuous actions as discrete bins)
            loss = F.mse_loss(torch.max(current_q, dim=1)[0], target_q)
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Update target network
        if timestep % target_update_freq == 0:
            target_network.load_state_dict(q_network.state_dict())
        
        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # Logging
        if timestep % 100 == 0:
            rewards.append(reward)
            users_served_history.append(info['users_served'])
            power_efficiency_history.append(info['power_efficiency'])
            
            if timestep % 10000 == 0:
                avg_reward = np.mean(rewards[-100:]) if rewards else 0
                avg_users = np.mean(users_served_history[-100:]) if users_served_history else 0
                print(f"Timestep: {timestep}, Avg Reward: {avg_reward:.3f}, "
                      f"Avg Users Served: {avg_users:.1f}, Epsilon: {epsilon:.3f}")
        
        # Reset if needed
        if done:
            state = env.reset()
        else:
            state = next_state
    
    # Save results
    results = {
        'rewards': rewards,
        'users_served': users_served_history,
        'power_efficiency': power_efficiency_history,
        'model': q_network.state_dict()
    }
    
    return results, q_network


def plot_results(results):
    """Generate comparison plots"""
    # Create plots directory
    os.makedirs('plots/dqn_baseline', exist_ok=True)
    
    # Smooth data for plotting
    def smooth(data, window=100):
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    # Plot 1: Reward over time
    plt.figure(figsize=(10, 6))
    plt.plot(smooth(results['rewards']), label='DQN Reward', color='red', alpha=0.7)
    plt.xlabel('Training Steps (x100)')
    plt.ylabel('Reward')
    plt.title('DQN Training Progress - Reward')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('plots/dqn_baseline/reward_progress.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Users served
    plt.figure(figsize=(10, 6))
    plt.plot(smooth(results['users_served']), label='DQN Users Served', color='blue', alpha=0.7)
    plt.axhline(y=65, color='green', linestyle='--', label='Target (65 users)')
    plt.xlabel('Training Steps (x100)')
    plt.ylabel('Users Served')
    plt.title('DQN Training Progress - Users Served')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('plots/dqn_baseline/users_served.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Power efficiency
    plt.figure(figsize=(10, 6))
    plt.plot(smooth(results['power_efficiency']), label='DQN Power Efficiency', color='orange', alpha=0.7)
    plt.xlabel('Training Steps (x100)')
    plt.ylabel('Users Served per Watt')
    plt.title('DQN Training Progress - Power Efficiency')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('plots/dqn_baseline/power_efficiency.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print final statistics
    print("\n=== DQN Final Results ===")
    print(f"Final Average Reward: {np.mean(results['rewards'][-100:]):.3f}")
    print(f"Final Average Users Served: {np.mean(results['users_served'][-100:]):.1f}")
    print(f"Final Power Efficiency: {np.mean(results['power_efficiency'][-100:]):.3f}")


if __name__ == "__main__":
    print("Starting DQN baseline training...")
    print(f"Device: {device}")
    
    # Train model
    results, model = train_dqn()
    
    # Generate plots
    plot_results(results)
    
    # Save model
    torch.save(model.state_dict(), 'models/dqn_baseline.pth')
    
    print("\nTraining complete! Check plots/dqn_baseline/ for results.")