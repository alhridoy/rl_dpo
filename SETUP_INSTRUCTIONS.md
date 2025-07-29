# UAV Constrained PPO Training Setup Instructions

## Overview
This project implements a complete UAV training system with all the requested requirements:
- ✅ 1 million timesteps training
- ✅ 1000 timesteps per episode  
- ✅ 0.1 second time granularity
- ✅ User distribution changes every 100 episodes
- ✅ FAA 1km altitude constraint with penalties
- ✅ Testing with different user distributions
- ✅ UAV trajectory visualization from start to end position
- ✅ Training/testing curves for rewards and power efficiency

## Installation Options

### Option 1: Full ML Training (Recommended)

For complete PPO GAE training with Stable-Baselines3:

```bash
# Create a new environment with Python 3.10 or 3.11 (required for PyTorch)
conda create -n uav_training python=3.10
conda activate uav_training

# Install PyTorch
conda install pytorch torchvision -c pytorch

# Install other dependencies
pip install stable-baselines3 gymnasium matplotlib tensorboard

# Run the full training
python train_uav_constrained.py
```

### Option 2: Using Current Environment (Demonstration Mode)

If you have Python 3.13 or other environments where PyTorch isn't available:

```bash
# Current setup already works for demonstration
python train_uav_constrained.py
```

This will run in simulation mode and demonstrate all features while generating visualizations.

## Training Configuration

The training script is configured with:

- **Algorithm**: PPO with GAE (λ=0.98)
- **Total timesteps**: 1,000,000
- **Episodes**: 1,000 (1000 timesteps each)
- **Network**: [512, 512, 256] hidden layers
- **Learning rate**: 0.0001
- **Batch size**: 1000
- **Epochs per update**: 10

## Key Features Implemented

### 1. Time Configuration
- Time granularity: 0.1 seconds
- Episode length: 1000 timesteps (100 seconds)

### 2. Constraints & Penalties
- **Altitude**: FAA 1km limit (1000m max) with penalties
- **Velocity**: 3 m/s maximum
- **Acceleration**: 1 m/s² maximum

### 3. User Distribution Changes
- Changes every 100 episodes to prevent overfitting
- 4 distribution types: uniform, clustered, edge, mixed

### 4. Testing & Visualization
- Automatic testing on all 4 user distributions
- 3D trajectory visualization showing UAV path over users
- Training curves for rewards and power efficiency
- Altitude and velocity constraint monitoring

## Output Files

When training completes, you'll get:

- `training_results_constrained.png` - Training performance curves
- `test_scenarios_results_constrained.png` - Testing results on different distributions
- `final_trajectory_visualization.png` - Final UAV trajectory with users
- `trajectory_plots/` - Episode-by-episode trajectory visualizations
- `ppo_constrained_uav_model_*` - Saved trained model

## Troubleshooting

### PyTorch Installation Issues
- Use Python 3.10 or 3.11 for best compatibility
- Try conda instead of pip for PyTorch installation
- For Apple Silicon Macs: `conda install pytorch::pytorch torchvision -c pytorch`

### Memory Issues
- Reduce batch_size from 1000 to 500 or 250
- Reduce network size from [512, 512, 256] to [256, 256, 128]

### Training Time
- Full training takes several hours
- Use callback frequency settings to monitor progress
- TensorBoard logs are saved for monitoring

## Verification

The script automatically verifies all requirements:
- ✅ Checks altitude violations and applies penalties
- ✅ Monitors velocity constraints  
- ✅ Tracks user distribution changes
- ✅ Generates comprehensive visualizations
- ✅ Tests on multiple scenarios post-training

## Performance Targets

- Users served: 85+ out of 100
- Power efficiency: 90%+
- Constraint violations: Minimized through penalties
- Convergence: Typically within 500-800 episodes