# UAV Reinforcement Learning Project

This project implements a reinforcement learning system to optimize the positioning and power allocation of a UAV (Unmanned Aerial Vehicle) serving as a wireless communication base station for ground users.

## Project Overview

The system uses **PPO (Proximal Policy Optimization)** to train a UAV to:
- **Navigate in 3D space** to optimize coverage
- **Allocate transmission power** across users and communication channels
- **Maximize the number of users served** while meeting minimum data rate requirements

## Key Features

- **100 ground users** distributed across a 1000x1000m area
- **5 communication subchannels** for power allocation
- **Realistic air-to-ground channel model** with line-of-sight probability
- **3D trajectory visualization** showing UAV movement patterns
- **Adaptive training** with user distribution changes to prevent overfitting

## Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Test the environment:
```bash
python test_environment.py
```

## Usage

### Running the Training

```bash
python rl.py
```

This will:
- Train the PPO agent for 1,000,000 timesteps (1000 episodes)
- Display power allocations and performance statistics
- Generate 3D trajectory plots showing UAV movement
- Save tensorboard logs for monitoring training progress

### Monitoring Training

View training progress with tensorboard:
```bash
tensorboard --logdir ./ppo_uav_tensorboard
```

## Files

- `rl.py` - Main training script with UAV environment and PPO training
- `Training_Hridoy.ipynb` - Jupyter notebook version (has the same bug fixes)
- `test_environment.py` - Simple test script to verify the environment works
- `requirements.txt` - Required Python packages
- `README.md` - This documentation

## Environment Details

### Action Space
- **Movement parameters** (6 values): acceleration magnitude and direction with stochastic control
- **Power allocation** (1000 values): power distribution across 100 users × 5 subchannels × 2 parameters (mean/variance)

### Observation Space
- UAV 3D position (x, y, z)
- UAV 3D velocity
- Channel gains to all 100 users

### Reward System
The UAV receives a reward equal to the number of users successfully served (users with data throughput ≥ 3.0 units).

## Bugs Fixed

1. **Missing `policy_kwargs`** - Added proper neural network architecture definition
2. **Reset method** - Fixed to handle `options` parameter for position reset control
3. **Throughput calculation** - Now properly incorporates power allocation
4. **Statistics tracking** - Fixed user served and throughput tracking
5. **Error handling** - Added checks for uninitialized variables
6. **Output formatting** - Limited verbose output to prevent spam

## Training Parameters

- **Learning rate**: 0.0001
- **Batch size**: 1000
- **Steps per update**: 10000
- **Discount factor (γ)**: 0.99
- **GAE lambda**: 0.98
- **Clip range**: 0.2
- **Entropy coefficient**: 0.01

## Expected Results

The trained UAV should learn to:
- Position itself optimally to serve the maximum number of users
- Allocate power efficiently across users and subchannels
- Adapt to changing user distributions
- Navigate in 3D space while maintaining communication quality

Training typically shows improvement in the number of users served over time, with the UAV learning more sophisticated movement and power allocation strategies.
