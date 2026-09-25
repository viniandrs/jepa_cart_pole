# Data Preparation Module

This module handles data collection and preprocessing for the JEPA CartPole project.

## Structure

```
data_prep/
├── __init__.py          # Module exports
├── environment.py       # Environment creation and configuration
├── collector.py         # Parallel data collection pipeline
├── merge.py            # Episode consolidation utilities
└── visualization.py    # Dataset visualization tools
```

## Key Components

### Environment (`environment.py`)
- `create_cartpole_env()`: Creates CartPole environment with pixel observations
- `setup_pygame_headless()`: Configures headless rendering
- `get_observation_shape()`: Returns observation dimensions

### Collector (`collector.py`)
- `collect_data()`: Main function for parallel episode collection
- `collect_episode()`: Collects a single episode
- `collect_data_worker()`: Worker process for parallel collection

### Merge (`merge.py`)
- `merge_episode_files()`: Consolidates temporary files into single HDF5
- `get_dataset_info()`: Retrieves dataset statistics

### Visualization (`visualization.py`)
- `plot_random_samples()`: Display random frames from dataset
- `plot_episode_sequence()`: Show temporal sequence from episode
- `plot_dataset_statistics()`: Statistical analysis plots
- `print_dataset_summary()`: Comprehensive dataset summary

## Usage Example

```python
from jepacartpole.data_prep import (
    collect_data,
    merge_episode_files,
    plot_random_samples
)

# Collect data (uses all available CPUs by default)
collect_data(
    num_episodes=500,
    max_steps=500,
    output_dir='../data/raw',
    action_interval=20,
    num_workers=None,  # None = use all CPUs
    chunk_size=None    # None = auto-calculate for smooth progress
)

# Merge into single file
merge_episode_files(
    episodes_dir='../data/raw',
    output_file='../data/vae/vae_data.h5',
    max_steps=500
)

# Visualize
plot_random_samples('../data/vae/vae_data.h5', num_samples=9)
```

## Performance Notes

The data collection uses **chunked parallel processing** with **real-time per-worker tracking**:

### Features
- **All CPUs utilized**: Uses all available CPU cores by default
- **Chunked work distribution**: Work split into small chunks (default: 10 episodes)
- **Real-time progress tracking**: See exact episode counts per worker
- **Live progress bar**: Updates as each episode completes across all workers

### Progress Display

```
Collecting episodes: |████████| 500/500 [02:30<00:00] W0:252 | W1:248

✅ Data collection completed!
   Total episodes collected: 500
   Episodes by worker:
      Worker 0: 252 episodes
      Worker 1: 248 episodes
```

**Progress bar format:**
- Total episodes completed / Total episodes
- Time elapsed and estimated time remaining
- `W0:X | W1:Y` - Episodes completed by each worker in real-time

### Performance on 2-CPU System (500 episodes)
- Sequential: ~5 minutes (1 worker)
- Parallel: ~2.5 minutes (2 workers)
- **Speedup**: 2x faster with full visibility into each worker's progress!

## Dataset Structure

The consolidated HDF5 dataset has the following structure:

```
vae_data.h5
├── images:  uint8   (num_episodes, max_steps, 400, 600, 3)
├── actions: float32 (num_episodes, max_steps)
├── rewards: float32 (num_episodes, max_steps)
└── dones:   bool    (num_episodes, max_steps)
```
