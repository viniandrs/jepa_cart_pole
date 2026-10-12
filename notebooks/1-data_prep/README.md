# Stage 1: Data Collection and Preprocessing

This directory contains the first stage of the JEPA CartPole project: collecting and preprocessing data from the CartPole-v1 environment.

## Notebooks

### 1-data_collection.ipynb
Main notebook for collecting CartPole episodes and creating the consolidated dataset.

**What it does:**
- Introduces the CartPole-v1 environment and pixel observations
- Shows the World Models-style preprocessing (crop + resize to 64×64)
- Runs parallel data collection using multiprocessing (one compressed `.npz` per episode in `data/raw/`)
- Consolidates the episode files into a single HDF5 dataset
- Visualizes random samples, episode sequences, and dataset statistics
- Saves results and visualizations to `results/data_prep/`

**Output:**
- `data/vae/vae_data.h5` - Consolidated dataset (500 episodes × 500 steps)
- `results/data_prep/random_samples.png` - Random frame samples
- `results/data_prep/episode_sequence.png` - Temporal sequence visualization
- `results/data_prep/dataset_statistics.png` - Statistical analysis plots

## Code Organization

All data collection logic is organized in `jepacartpole/data_prep/`:

```
jepacartpole/data_prep/
├── __init__.py          # Module exports
├── environment.py       # Environment creation and frame preprocessing
├── collector.py         # Parallel data collection
├── merge.py             # Dataset consolidation into one HDF5 file
└── visualization.py    # Visualization tools
```

The notebook imports and uses these modules, keeping the notebook clean and focused on:
1. Explaining what we're doing and why
2. Running the pipeline
3. Visualizing and analyzing results

## Usage

Open and run `1-data_collection.ipynb` in Jupyter. The notebook will:

1. Set up the environment and imports
2. Demonstrate CartPole pixel observations
3. Collect episodes using parallel workers (re-running resumes an interrupted collection)
4. Merge into a single dataset file
5. Create comprehensive visualizations

All parameters (number of episodes, steps per episode, etc.) are configurable in the notebook.

## Next Steps

After completing this notebook, proceed to:
- **Stage 2**: `notebooks/2-vae/1-vae_training.ipynb` for VAE encoder training
