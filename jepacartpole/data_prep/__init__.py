"""
Data collection and preprocessing module.

This module contains code for collecting CartPole episodes and preparing
datasets for VAE and JEPA training.
"""

from .environment import (
    setup_pygame_headless,
    create_cartpole_env,
    get_observation_shape
)

from .collector import (
    collect_data,
    collect_episode,
    collect_data_chunk
)

from .merge import (
    merge_episode_files,
    get_dataset_info,
    check_dataset_exists
)

from .consolidate import (
    consolidate_episodes_in_chunks,
    get_chunk_info
)

from .chunked_dataset import (
    ChunkedHDF5Dataset,
    create_chunked_dataset
)

from .visualization import (
    plot_random_samples,
    plot_episode_sequence,
    plot_dataset_statistics,
    print_dataset_summary
)

__all__ = [
    # Environment
    'setup_pygame_headless',
    'create_cartpole_env',
    'get_observation_shape',
    # Collection
    'collect_data',
    'collect_episode',
    'collect_data_chunk',
    # Merging
    'merge_episode_files',
    'get_dataset_info',
    'check_dataset_exists',
    # Consolidation (chunked)
    'consolidate_episodes_in_chunks',
    'get_chunk_info',
    'ChunkedHDF5Dataset',
    'create_chunked_dataset',
    # Visualization
    'plot_random_samples',
    'plot_episode_sequence',
    'plot_dataset_statistics',
    'print_dataset_summary',
]
