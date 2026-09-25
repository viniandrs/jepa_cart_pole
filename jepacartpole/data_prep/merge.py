"""
Episode consolidation utilities for merging individual HDF5 files.
"""

import os
import h5py
from tqdm import tqdm


def merge_episode_files(episodes_dir, output_file, max_steps=500,
                       delete_temp_files=True, verbose=True):
    """
    Merge multiple episode HDF5 files into a single consolidated dataset.

    The consolidated dataset has the structure:
        images:  uint8   (num_episodes, max_steps, 400, 600, 3)
        actions: float32 (num_episodes, max_steps)
        rewards: float32 (num_episodes, max_steps)
        dones:   bool    (num_episodes, max_steps)

    Args:
        episodes_dir (str): Directory containing individual episode files
        output_file (str): Path for the consolidated output file
        max_steps (int): Expected number of steps per episode
        delete_temp_files (bool): If True, delete temporary episode files after merging
        verbose (bool): If True, print progress messages

    Returns:
        dict: Statistics about the merged dataset (num_episodes, max_steps, file_size)
    """
    # Step 1: Locate all episode files
    episode_files = [
        os.path.join(episodes_dir, f)
        for f in os.listdir(episodes_dir)
        if f.endswith('.h5') or f.endswith('.hdf5')
    ]

    num_episodes = len(episode_files)
    if num_episodes == 0:
        raise ValueError(f"No HDF5 episode files found in directory: {episodes_dir}")

    if verbose:
        print(f"Found {num_episodes} episode files in '{episodes_dir}'.")

    # Sort files for consistent ordering
    episode_files.sort()

    # Step 2: Verify episode integrity and determine image shape
    with h5py.File(episode_files[0], 'r') as h5f:
        first_image = h5f['images'][0]
        image_shape = first_image.shape  # (height, width, channels)

    for file in episode_files:
        with h5py.File(file, 'r') as h5f:
            actions_shape = h5f['actions'].shape
            if actions_shape[0] != max_steps:
                raise ValueError(
                    f"Episode file {file} has {actions_shape[0]} steps, expected {max_steps}."
                )

    if verbose:
        print(f"Image shape: {image_shape}")
        print(f"Initializing consolidated dataset...")

    # Step 3: Initialize the merged HDF5 file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with h5py.File(output_file, 'w') as merged_h5f:
        # Initialize datasets with optimized chunking
        actions_shape = (num_episodes, max_steps)
        dones_shape = (num_episodes, max_steps)
        images_shape = (num_episodes, max_steps, *image_shape)
        rewards_shape = (num_episodes, max_steps)

        merged_h5f.create_dataset('actions', shape=actions_shape,
                                 dtype='float32', chunks=(1, max_steps))
        merged_h5f.create_dataset('dones', shape=dones_shape,
                                 dtype='bool', chunks=(1, max_steps))
        merged_h5f.create_dataset('images', shape=images_shape,
                                 dtype='uint8', chunks=(1, max_steps, *image_shape))
        merged_h5f.create_dataset('rewards', shape=rewards_shape,
                                 dtype='float32', chunks=(1, max_steps))

        # Step 4: Iterate through each episode and write to the merged file
        iterator = tqdm(episode_files, desc="Merging Episodes") if verbose else episode_files

        for idx, episode_file in enumerate(iterator):
            with h5py.File(episode_file, 'r') as ep_h5f:
                # Read datasets from the episode file
                actions = ep_h5f['actions'][:]
                dones = ep_h5f['dones'][:]
                images = ep_h5f['images'][:]
                rewards = ep_h5f['rewards'][:]

                # Write to the merged datasets
                merged_h5f['actions'][idx] = actions
                merged_h5f['dones'][idx] = dones
                merged_h5f['images'][idx] = images
                merged_h5f['rewards'][idx] = rewards

    # Step 5: Clean up temporary files if requested
    if delete_temp_files:
        for episode_file in episode_files:
            try:
                os.remove(episode_file)
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not delete temporary file '{episode_file}': {e}")

    # Calculate file size
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)

    if verbose:
        print(f"Merging completed successfully.")
        print(f"Dataset saved to: {output_file}")
        print(f"File size: {file_size_mb:.2f} MB")

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
