# Data Preparation Module

This module handles data collection and preprocessing for the JEPA CartPole project. Following World Models (Ha & Schmidhuber, 2018), frames are cropped and resized to 64×64, each episode is saved as a compressed `.npz` file, and the episodes are then consolidated into a single HDF5 dataset.

## Structure

```
data_prep/
├── __init__.py          # Module exports
├── environment.py       # Environment creation and frame preprocessing
├── collector.py         # Parallel episode collection (.npz per episode)
├── merge.py             # Consolidation of episodes into one HDF5 file
└── visualization.py     # Dataset visualization tools
```

## Key Components

### Environment (`environment.py`)
- `create_cartpole_env(grayscale=False, preprocess=True)`: CartPole-v1 with pixel observations, preprocessed to 64×64 by default
- `preprocess_frame()`: Crops rows `CROP_TOP:CROP_BOTTOM` (150–330) of the 400×600 render and resizes to `FRAME_SIZE` (64) with area interpolation
- `setup_pygame_headless()`: Configures headless rendering (call before creating environments)
- `get_observation_shape()`: Returns the preprocessed observation shape

### Collector (`collector.py`)
- `collect_data()`: Parallel episode collection over all CPUs, with per-worker progress. Episodes that already exist are skipped, so an interrupted run can be resumed
- `collect_episode()`: Collects a single episode (seeded with `seed + episode_id`) and writes `episode_XXXXX.npz`
- `collect_data_chunk()`: Worker process for parallel collection

### Merge (`merge.py`)
- `merge_episode_files()`: Consolidates the `.npz` episode files into a single HDF5 file, then deletes them
- `check_dataset_exists()`: Verifies if the final dataset already exists
- `get_dataset_info()`: Retrieves dataset statistics

### Visualization (`visualization.py`)
- `plot_random_samples()`: Display random frames from dataset
- `plot_episode_sequence()`: Show temporal sequence from episode
- `plot_dataset_statistics()`: Statistical analysis plots
- `print_dataset_summary()`: Comprehensive dataset summary

## Usage

```python
from jepacartpole.data_prep import collect_data, merge_episode_files
from jepacartpole.utils import get_data_dir

collect_data(
    num_episodes=500,
    max_steps=500,
    output_dir=get_data_dir('raw'),
)

merge_episode_files(
    episodes_dir=get_data_dir('raw'),
    output_file=get_data_dir('vae') / 'vae_data.h5',
    max_steps=500,
)
```

## Dataset Structure

```
vae_data.h5
├── images:  uint8   (num_episodes, max_steps, 64, 64, 3)
├── actions: float32 (num_episodes, max_steps)
├── rewards: float32 (num_episodes, max_steps)
└── dones:   bool    (num_episodes, max_steps)
attrs: crop_rows = (150, 330), frame_size = 64
```

`dones` marks steps where the pole fell; the environment is reset and the episode continues until `max_steps`. 500 episodes × 500 steps is about 3–4 GB.

## See Also

- `notebooks/1-data_prep/1-data_collection.ipynb` - Complete workflow example
