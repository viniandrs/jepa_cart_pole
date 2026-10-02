"""
Episode consolidation utilities for merging individual episode files
into a single HDF5 dataset.
"""

import os
import h5py
import numpy as np
from tqdm import tqdm
from .environment import CROP_TOP, CROP_BOTTOM, FRAME_SIZE


def merge_episode_files(episodes_dir, output_file, max_steps=500,
                       delete_temp_files=True, verbose=True):
    """
    Merge the per-episode .npz files into a single consolidated HDF5 dataset.

    The consolidated dataset has the structure:
        images:  uint8   (num_episodes, max_steps, 64, 64, C)
        actions: float32 (num_episodes, max_steps)
        rewards: float32 (num_episodes, max_steps)
        dones:   bool    (num_episodes, max_steps)

    The preprocessing parameters (crop rows and frame size) are stored as
    file attributes.

    Args:
        episodes_dir (str): Directory containing individual episode files
        output_file (str): Path for the consolidated output file
        max_steps (int): Expected number of steps per episode
        delete_temp_files (bool): If True, delete each episode file right after
                                  it has been written to the merged dataset
        verbose (bool): If True, print progress messages

    Returns:
        dict: Statistics about the merged dataset (num_episodes, max_steps, file_size)
    """
    episodes_dir = str(episodes_dir)
    output_file = str(output_file)

    # Step 1: Locate all episode files (sorted for consistent ordering)
    episode_files = sorted(
        os.path.join(episodes_dir, f)
        for f in os.listdir(episodes_dir)
        if f.endswith('.npz')
    )

    num_episodes = len(episode_files)
    if num_episodes == 0:
        raise ValueError(f"No .npz episode files found in directory: {episodes_dir}")

    if verbose:
        print(f"Found {num_episodes} episode files in '{episodes_dir}'.")

    # Step 2: Verify episode integrity and determine image shape
    with np.load(episode_files[0]) as ep:
        image_shape = ep['images'].shape[1:]  # (height, width, channels)

    for file in episode_files:
        with np.load(file) as ep:
            num_steps = ep['actions'].shape[0]
        if num_steps != max_steps:
            raise ValueError(
                f"Episode file {file} has {num_steps} steps, expected {max_steps}."
            )

    if verbose:
        print(f"Image shape: {image_shape}")
        print(f"Initializing consolidated dataset...")

    # Step 3: Initialize the merged HDF5 file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with h5py.File(output_file, 'w') as merged_h5f:
        # One HDF5 chunk per episode so whole episodes are read efficiently
        merged_h5f.create_dataset('images', shape=(num_episodes, max_steps, *image_shape),
                                 dtype='uint8', chunks=(1, max_steps, *image_shape))
        merged_h5f.create_dataset('actions', shape=(num_episodes, max_steps),
                                 dtype='float32', chunks=(1, max_steps))
        merged_h5f.create_dataset('rewards', shape=(num_episodes, max_steps),
                                 dtype='float32', chunks=(1, max_steps))
        merged_h5f.create_dataset('dones', shape=(num_episodes, max_steps),
                                 dtype='bool', chunks=(1, max_steps))

        merged_h5f.attrs['crop_rows'] = (CROP_TOP, CROP_BOTTOM)
        merged_h5f.attrs['frame_size'] = FRAME_SIZE

        # Step 4: Iterate through each episode and write to the merged file
        iterator = tqdm(episode_files, desc="Merging Episodes") if verbose else episode_files

        for idx, episode_file in enumerate(iterator):
            with np.load(episode_file) as ep:
                for key in ('images', 'actions', 'rewards', 'dones'):
                    merged_h5f[key][idx] = ep[key]

    # Step 5: Clean up episode files only once the merged file is complete
    if delete_temp_files:
        for episode_file in episode_files:
            try:
                os.remove(episode_file)
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not delete temporary file '{episode_file}': {e}")

    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Merging completed successfully!")
        print(f"{'='*60}")
        print(f"Dataset saved to: {output_file}")
        print(f"File size: {file_size_mb:.2f} MB")
        print(f"Episodes: {num_episodes}")
        print(f"Steps per episode: {max_steps}")
        print(f"Total frames: {num_episodes * max_steps:,}")
        print(f"{'='*60}")

    return {
        'num_episodes': num_episodes,
        'max_steps': max_steps,
        'image_shape': image_shape,
        'file_size_mb': file_size_mb,
        'output_file': output_file
    }


def get_dataset_info(h5_path):
    """
    Get information about a consolidated HDF5 dataset.

    Args:
        h5_path (str): Path to the HDF5 file

    Returns:
        dict: Dataset information including shapes and dtypes
    """
    with h5py.File(h5_path, 'r') as h5f:
        info = {
            'file_path': h5_path,
            'file_size_mb': os.path.getsize(h5_path) / (1024 * 1024),
            'datasets': {}
        }

        for key in h5f.keys():
            dataset = h5f[key]
            info['datasets'][key] = {
                'shape': dataset.shape,
                'dtype': dataset.dtype,
                'size_mb': dataset.nbytes / (1024 * 1024)
            }

        # Calculate derived statistics
        if 'images' in h5f:
            num_episodes, max_steps = h5f['images'].shape[:2]
            total_frames = num_episodes * max_steps
            info['num_episodes'] = num_episodes
            info['max_steps'] = max_steps
            info['total_frames'] = total_frames

    return info


def check_dataset_exists(output_file, verbose=True):
    """
    Check if the final consolidated dataset already exists.

    Args:
        output_file (str): Path to the consolidated dataset file
        verbose (bool): If True, print status message

    Returns:
        bool: True if dataset exists and is valid, False otherwise
    """
    if not os.path.exists(output_file):
        return False

    try:
        # Try to open and verify the file
        with h5py.File(output_file, 'r') as h5f:
            # Check if required datasets exist
            required_keys = ['images', 'actions', 'rewards', 'dones']
            if all(key in h5f for key in required_keys):
                if verbose:
                    num_episodes = h5f['images'].shape[0]
                    max_steps = h5f['images'].shape[1]
                    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)

                    print(f"✅ Found existing dataset: {output_file}")
                    print(f"   Episodes: {num_episodes}")
                    print(f"   Steps per episode: {max_steps}")
                    print(f"   File size: {file_size_mb:.2f} MB")

                return True
            else:
                if verbose:
                    print(f"⚠️  File exists but is missing required datasets")
                return False

    except Exception as e:
        if verbose:
            print(f"⚠️  File exists but appears corrupted: {e}")
        return False
