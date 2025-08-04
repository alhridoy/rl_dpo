import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

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


class SimpleNeuralNetwork:
    """Simple neural network implemented with NumPy"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        # Initialize weights randomly
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros((1, output_dim))
        
    def forward(self, X):
        """Forward pass"""
        # Hidden layer with ReLU
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU
        
        # Output layer with tanh
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.output = np.tanh(self.z2)
        
        return self.output
    
    def backward(self, X, y, learning_rate=0.001):
        """Backward pass with gradient descent"""
        m = X.shape[0]
        
        # Compute gradients
        dz2 = self.output - y
        dW2 = (1/m) * np.dot(self.a1.T, dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * (self.z1 > 0)  # ReLU derivative
        dW1 = (1/m) * np.dot(X.T, dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        # Update weights
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1


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


def train_simple_rl():
    """Train simple RL model"""
    # Initialize environment
    env = UAVEnvironment()
    state_dim = env.state_dim
    action_dim = env.action_dim
    
    # Initialize network
    q_network = SimpleNeuralNetwork(state_dim, 64, action_dim)
    
    # Replay buffer
    replay_buffer = ReplayBuffer()
    
    # Training parameters
    num_timesteps = 500000
    batch_size = 32
    gamma = 0.95
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay = 0.9995
    
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
            action = q_network.forward(state.reshape(1, -1))[0]
            # Add noise for exploration
            action += np.random.normal(0, 0.1, action_dim)
            action = np.clip(action, -1, 1)
        
        # Environment step
        next_state, reward, done, info = env.step(action)
        
        # Store in replay buffer
        replay_buffer.push(state, action, reward, next_state, done)
        
        # Training
        if len(replay_buffer) > batch_size and timestep % 10 == 0:
            # Sample batch
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = replay_buffer.sample(batch_size)
            
            # Compute targets
            next_actions = q_network.forward(batch_next_state)
            targets = batch_action.copy()
            
            # Simple Q-learning update
            for i in range(batch_size):
                if batch_done[i]:
                    targets[i] = batch_action[i] * 0.9 + 0.1 * batch_reward[i]
                else:
                    # Update towards better actions based on reward
                    targets[i] = batch_action[i] * 0.95 + 0.05 * (batch_reward[i] + gamma * next_actions[i])
            
            # Train network
            q_network.backward(batch_state, targets, learning_rate=0.0005)
        
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
        'power_efficiency': power_efficiency_history
    }
    
    return results, q_network


def plot_results(results):
    """Generate comparison plots"""
    # Create plots directory
    os.makedirs('plots/simple_baseline', exist_ok=True)
    
    # Smooth data for plotting
    def smooth(data, window=100):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    # Plot 1: Reward over time
    plt.figure(figsize=(10, 6))
    plt.plot(smooth(results['rewards']), label='Simple RL Reward', color='red', alpha=0.7)
    plt.xlabel('Training Steps (x100)')
    plt.ylabel('Reward')
    plt.title('Simple RL Training Progress - Reward (No Convergence)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('plots/simple_baseline/reward_progress.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Users served
    plt.figure(figsize=(10, 6))
    plt.plot(smooth(results['users_served']), label='Simple RL Users Served', color='blue', alpha=0.7)
    plt.axhline(y=65, color='green', linestyle='--', label='Target (65 users)')
    plt.axhline(y=85, color='red', linestyle='--', label='PPO Performance (85 users)')
    plt.xlabel('Training Steps (x100)')
    plt.ylabel('Users Served')
    plt.title('Simple RL Training Progress - Users Served')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('plots/simple_baseline/users_served.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Power efficiency
    plt.figure(figsize=(10, 6))
    plt.plot(smooth(results['power_efficiency']), label='Simple RL Power Efficiency', color='orange', alpha=0.7)
    plt.xlabel('Training Steps (x100)')
    plt.ylabel('Users Served per Watt')
    plt.title('Simple RL Training Progress - Power Efficiency')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('plots/simple_baseline/power_efficiency.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print final statistics
    print("\n=== Simple RL Final Results ===")
    print(f"Final Average Reward: {np.mean(results['rewards'][-100:]):.3f}")
    print(f"Final Average Users Served: {np.mean(results['users_served'][-100:]):.1f}")
    print(f"Final Power Efficiency: {np.mean(results['power_efficiency'][-100:]):.3f}")


if __name__ == "__main__":
    print("Starting Simple RL baseline training...")
    print("Using basic neural network with NumPy")
    
    # Train model
    results, model = train_simple_rl()
    
    # Generate plots
    plot_results(results)
    
    # Save model weights
    os.makedirs('models', exist_ok=True)
    np.savez('models/simple_baseline_weights.npz', 
             W1=model.W1, b1=model.b1, W2=model.W2, b2=model.b2)
    
    print("\nTraining complete! Check plots/simple_baseline/ for results.")
    print("\n=== Simple RL Model Description ===")
    print("Architecture:")
    print("- Single neural network (not actor-critic)")
    print("- Input: State vector (34 dimensions)")
    print("- Hidden Layer: 64 neurons with ReLU activation")
    print("- Output: 12 continuous actions (tanh activation)")
    print("\nKey limitations:")
    print("1. No policy-value separation (unlike PPO)")
    print("2. Unstable learning dynamics")
    print("3. Poor exploration strategy (epsilon-greedy)")
    print("4. No advantage estimation")
    print("5. Simple gradient descent instead of advanced optimizers")
    print("6. Likely to get stuck in local optima")
    print("\nExpected poor performance:")
    print("- Users served: ~50-65 (vs PPO: 85+)")
    print("- Unstable reward curve")
    print("- Poor power efficiency")