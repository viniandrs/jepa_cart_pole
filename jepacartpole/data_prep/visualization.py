"""
Visualization utilities for CartPole dataset exploration.
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path


def plot_random_samples(h5_path, num_samples=9, figsize=(15, 10), save_path=None):
    """
    Display random images from the dataset in a grid.

    Args:
        h5_path (str): Path to the HDF5 dataset
        num_samples (int): Number of random samples to display
        figsize (tuple): Figure size
        save_path (str, optional): Path to save the figure

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    with h5py.File(h5_path, 'r') as h5f:
        images = h5f['images']
        num_episodes, max_steps = images.shape[:2]
        total_frames = num_episodes * max_steps

        # Sample random indices
        random_indices = np.random.choice(total_frames, num_samples, replace=False)

        # Calculate grid dimensions
        cols = int(np.ceil(np.sqrt(num_samples)))
        rows = int(np.ceil(num_samples / cols))

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten() if num_samples > 1 else [axes]

        for idx, frame_idx in enumerate(random_indices):
            episode = frame_idx // max_steps
            step = frame_idx % max_steps

            # Load the image
            img = images[episode, step]

            # Plot
            axes[idx].imshow(img)
            axes[idx].set_title(f'Episode {episode}, Step {step}', fontsize=10)
            axes[idx].axis('off')

        # Hide unused subplots
        for idx in range(num_samples, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")

        return fig


def plot_episode_sequence(h5_path, episode_idx=0, num_frames=10,
                         step_interval=None, figsize=(20, 4), save_path=None):
    """
    Display a sequence of frames from a single episode.

    Args:
        h5_path (str): Path to the HDF5 dataset
        episode_idx (int): Index of the episode to visualize
        num_frames (int): Number of frames to display
        step_interval (int, optional): Interval between frames. If None, evenly spaced
        figsize (tuple): Figure size
        save_path (str, optional): Path to save the figure

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    with h5py.File(h5_path, 'r') as h5f:
        images = h5f['images']
        actions = h5f['actions']
        rewards = h5f['rewards']
        dones = h5f['dones']

        num_episodes, max_steps = images.shape[:2]

        if episode_idx >= num_episodes:
            raise ValueError(f"Episode {episode_idx} out of range (max: {num_episodes-1})")

        # Determine frame indices
        if step_interval is None:
            # Evenly spaced frames
            frame_indices = np.linspace(0, max_steps - 1, num_frames, dtype=int)
        else:
            # Regular interval
            frame_indices = np.arange(0, max_steps, step_interval)[:num_frames]

        fig, axes = plt.subplots(1, len(frame_indices), figsize=figsize)
        if len(frame_indices) == 1:
            axes = [axes]

        for idx, step in enumerate(frame_indices):
            img = images[episode_idx, step]
            action = actions[episode_idx, step]
            reward = rewards[episode_idx, step]
            done = dones[episode_idx, step]

            axes[idx].imshow(img)
            action_label = 'Left' if action < 0.5 else 'Right'
            title = f'Step {step}\nAction: {action_label}\nReward: {reward:.1f}'
            if done:
                title += '\n(Done)'
            axes[idx].set_title(title, fontsize=9)
            axes[idx].axis('off')

        plt.suptitle(f'Episode {episode_idx} Sequence', fontsize=14, y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")

        return fig


def plot_dataset_statistics(h5_path, figsize=(15, 5), save_path=None):
    """
    Plot statistical information about the dataset.

    Args:
        h5_path (str): Path to the HDF5 dataset
        figsize (tuple): Figure size
        save_path (str, optional): Path to save the figure

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    with h5py.File(h5_path, 'r') as h5f:
        actions = h5f['actions'][:]
        rewards = h5f['rewards'][:]
        dones = h5f['dones'][:]

        num_episodes, max_steps = actions.shape

        # Calculate statistics
        total_rewards_per_episode = rewards.sum(axis=1)
        actions_distribution = (actions > 0.5).sum() / actions.size
        dones_per_episode = dones.sum(axis=1)

        # Create plots
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Plot 1: Total rewards per episode
        axes[0].hist(total_rewards_per_episode, bins=30, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Total Reward')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'Reward Distribution Across Episodes\nMean: {total_rewards_per_episode.mean():.1f}')
        axes[0].grid(alpha=0.3)

        # Plot 2: Action distribution
        actions_counts = [(actions <= 0.5).sum(), (actions > 0.5).sum()]
        axes[1].bar(['Left (0)', 'Right (1)'], actions_counts, edgecolor='black', alpha=0.7)
        axes[1].set_ylabel('Count')
        axes[1].set_title(f'Action Distribution\nRight: {actions_distribution*100:.1f}%')
        axes[1].grid(alpha=0.3, axis='y')

        # Plot 3: Done flags per episode
        axes[2].hist(dones_per_episode, bins=30, edgecolor='black', alpha=0.7)
        axes[2].set_xlabel('Number of "Done" Flags')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title(f'Episode Terminations\nMean: {dones_per_episode.mean():.1f}')
        axes[2].grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")

        return fig


def print_dataset_summary(h5_path):
    """
    Print a comprehensive summary of the dataset.

    Args:
        h5_path (str): Path to the HDF5 dataset
    """
    import os

    with h5py.File(h5_path, 'r') as h5f:
        print("=" * 60)
        print(f"Dataset Summary: {Path(h5_path).name}")
        print("=" * 60)

        # File information
        file_size_mb = os.path.getsize(h5_path) / (1024 * 1024)
        print(f"\nFile Size: {file_size_mb:.2f} MB")
        print(f"File Path: {h5_path}")

        # Dataset structure
        print("\nDataset Structure:")
        for key in h5f.keys():
            dataset = h5f[key]
            print(f"  {key:8s}: {str(dataset.dtype):8s} {dataset.shape}")

        # Statistics
        images = h5f['images']
        actions = h5f['actions'][:]
        rewards = h5f['rewards'][:]
        dones = h5f['dones'][:]

        num_episodes, max_steps = images.shape[:2]
        total_frames = num_episodes * max_steps

        print(f"\nEpisode Information:")
        print(f"  Number of episodes: {num_episodes}")
        print(f"  Steps per episode: {max_steps}")
        print(f"  Total frames: {total_frames:,}")

        print(f"\nReward Statistics:")
        print(f"  Mean reward per step: {rewards.mean():.3f}")
        print(f"  Total reward per episode (mean): {rewards.sum(axis=1).mean():.1f}")
        print(f"  Total reward per episode (std): {rewards.sum(axis=1).std():.1f}")

        print(f"\nAction Distribution:")
        left_actions = (actions <= 0.5).sum()
        right_actions = (actions > 0.5).sum()
        print(f"  Left (0): {left_actions:,} ({left_actions/actions.size*100:.1f}%)")
        print(f"  Right (1): {right_actions:,} ({right_actions/actions.size*100:.1f}%)")

        print(f"\nEpisode Terminations:")
        dones_per_episode = dones.sum(axis=1)
        print(f"  Mean terminations per episode: {dones_per_episode.mean():.2f}")
        print(f"  Max terminations per episode: {dones_per_episode.max()}")

        print("=" * 60)
