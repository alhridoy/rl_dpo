# Simple Deep RL Baseline Model Description

## Overview
This document describes the simple Deep Reinforcement Learning (DRL) baseline model used for comparison with the PPO approach in UAV resource allocation.

## Model Architecture

### Neural Network Structure
- **Type**: Single feed-forward neural network (not actor-critic)
- **Input Layer**: 34 dimensions
  - UAV states: 9 dimensions (3 UAVs × 3 features: x, y, height)
  - User density grid: 25 dimensions (5×5 grid)
- **Hidden Layer**: 64 neurons with ReLU activation
- **Output Layer**: 12 dimensions with tanh activation
  - 3 UAVs × 4 actions each (Δx, Δy, Δheight, power)

### Implementation Details
```python
class SimpleNeuralNetwork:
    def __init__(self, input_dim=34, hidden_dim=64, output_dim=12):
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros((1, output_dim))
```

## Training Algorithm

### Key Components
1. **Experience Replay Buffer**: Stores past experiences for batch training
2. **Epsilon-Greedy Exploration**: Random action selection with decaying probability
3. **Simple Q-Learning Update**: Basic temporal difference learning
4. **Batch Size**: 32
5. **Learning Rate**: 0.0005
6. **Discount Factor (γ)**: 0.95

### Update Rule
The model uses a simplified Q-learning approach where actions are updated based on:
```
target_action = current_action * 0.95 + 0.05 * (reward + γ * next_action)
```

## Key Limitations Compared to PPO

### 1. No Actor-Critic Architecture
- PPO uses separate networks for policy (actor) and value estimation (critic)
- This baseline uses a single network attempting to do both

### 2. Unstable Learning Dynamics
- No trust region or clipping mechanism
- Simple gradient descent prone to large policy changes
- No KL divergence constraints

### 3. Poor Exploration Strategy
- Epsilon-greedy is less sophisticated than PPO's stochastic policy
- No entropy bonus to encourage exploration
- Gets stuck in local optima easily

### 4. No Advantage Estimation
- PPO uses Generalized Advantage Estimation (GAE)
- This model has no concept of advantage, just raw rewards

### 5. Simplified Optimization
- Basic gradient descent vs PPO's advanced optimization
- No gradient clipping or normalization
- No mini-batch optimization over multiple epochs

## Expected Performance

### Metrics Comparison
| Metric | Simple RL | PPO |
|--------|-----------|-----|
| Users Served | ~4-65 | 85+ |
| Reward Convergence | Poor/None | Good |
| Power Efficiency | Low | High |
| Training Stability | Unstable | Stable |

### Performance Characteristics
1. **Users Served**: Typically serves only 4-65 users (vs 85+ for PPO)
2. **Reward**: Negative rewards, no convergence
3. **Power Efficiency**: Poor optimization of power allocation
4. **Learning Curve**: Erratic, often deteriorates over time

## Why This Model Fails

### 1. Credit Assignment Problem
Without proper value function estimation, the model cannot determine which actions led to good/bad outcomes.

### 2. High Variance
Single network trying to handle both policy and value leads to high variance in updates.

### 3. Exploration-Exploitation Imbalance
Epsilon-greedy doesn't adapt based on uncertainty, leading to either too much or too little exploration.

### 4. No Sample Efficiency
Unlike PPO which reuses data through multiple epochs, this model uses each sample only once.

## Conclusion
This simple Deep RL baseline demonstrates why advanced algorithms like PPO are necessary for complex multi-agent resource allocation problems. The lack of:
- Separate policy and value networks
- Stable optimization techniques
- Proper advantage estimation
- Effective exploration strategies

Results in poor performance that fails to solve the UAV resource allocation problem effectively.