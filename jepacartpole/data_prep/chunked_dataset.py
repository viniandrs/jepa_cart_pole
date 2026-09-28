"""
PyTorch Dataset for reading from chunked HDF5 files in S3.

This module provides a Dataset that can read from multiple chunk files,
either in S3 or locally, without loading everything into memory.
"""

import h5py
import s3fs
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, List


class ChunkedHDF5Dataset(Dataset):
    """
    PyTorch Dataset that reads from multiple HDF5 chunks.

    Supports both local and S3 storage. Lazy loads data on demand.

    Args:
        chunk_paths (List[str]): List of paths to chunk files
                                 Can be local paths or s3:// URLs
        use_s3 (bool): If True, paths are S3 URLs
        transform (callable, optional): Transform to apply to images

    Example:
        # From S3
        chunks = [
            'bucket/chunks/chunk_0000.h5',
            'bucket/chunks/chunk_0001.h5',
        ]
        dataset = ChunkedHDF5Dataset(chunks, use_s3=True)

        # From local
        chunks = [
            '/data/chunk_0000.h5',
            '/data/chunk_0001.h5',
        ]
        dataset = ChunkedHDF5Dataset(chunks, use_s3=False)

        # Use with DataLoader
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
    """

    def __init__(
        self,
        chunk_paths: List[str],
        use_s3: bool = False,
        transform: Optional[callable] = None
    ):
        self.chunk_paths = chunk_paths
        self.use_s3 = use_s3
        self.transform = transform

        if use_s3:
            self.fs = s3fs.S3FileSystem()
        else:
            self.fs = None

        # Build index: map global index → (chunk_idx, local_episode_idx, step_idx)
        self._build_index()

    def _build_index(self):
        """Build index mapping global idx to (chunk, episode, step)."""
        self.chunk_episodes = []  # Number of episodes per chunk
        self.chunk_steps = []     # Steps per episode (should be same for all)

        # Open each chunk to get dimensions
        for chunk_path in self.chunk_paths:
            if self.use_s3:
                with self.fs.open(chunk_path, 'rb') as f:
                    with h5py.File(f, 'r') as h5f:
                        num_episodes = h5f['images'].shape[0]
                        max_steps = h5f['images'].shape[1]
            else:
                with h5py.File(chunk_path, 'r') as h5f:
                    num_episodes = h5f['images'].shape[0]
                    max_steps = h5f['images'].shape[1]

            self.chunk_episodes.append(num_episodes)
            self.chunk_steps.append(max_steps)

        # Total frames
        self.total_episodes = sum(self.chunk_episodes)
        self.max_steps = self.chunk_steps[0]  # Assuming all chunks have same steps
        self.total_frames = self.total_episodes * self.max_steps

        # Cumulative episodes for quick lookup
        self.cumulative_episodes = [0]
        for num_ep in self.chunk_episodes:
            self.cumulative_episodes.append(
                self.cumulative_episodes[-1] + num_ep
            )

    def _get_chunk_for_episode(self, episode_idx):
        """Get chunk index and local episode index for a global episode index."""
        for chunk_idx in range(len(self.chunk_paths)):
            if episode_idx < self.cumulative_episodes[chunk_idx + 1]:
                local_episode = episode_idx - self.cumulative_episodes[chunk_idx]
                return chunk_idx, local_episode

        raise IndexError(f"Episode index {episode_idx} out of range")

    def __len__(self):
        """Return total number of frames (episodes * steps)."""
        return self.total_frames

    def __getitem__(self, idx):
        """
        Get a single frame.

        Args:
            idx (int): Global frame index

        Returns:
            tuple: (image, action, reward, done)
                   image: Tensor (C, H, W)
                   action: float
                   reward: float
                   done: bool
        """
        # Convert global frame index to (episode, step)
        episode_idx = idx // self.max_steps
        step_idx = idx % self.max_steps

        # Get chunk and local episode
        chunk_idx, local_episode = self._get_chunk_for_episode(episode_idx)

        # Read data from chunk
        chunk_path = self.chunk_paths[chunk_idx]

        if self.use_s3:
            with self.fs.open(chunk_path, 'rb') as f:
                with h5py.File(f, 'r') as h5f:
                    image = h5f['images'][local_episode, step_idx]
                    action = h5f['actions'][local_episode, step_idx]
                    reward = h5f['rewards'][local_episode, step_idx]
                    done = h5f['dones'][local_episode, step_idx]
        else:
            with h5py.File(chunk_path, 'r') as h5f:
                image = h5f['images'][local_episode, step_idx]
                action = h5f['actions'][local_episode, step_idx]
                reward = h5f['rewards'][local_episode, step_idx]
                done = h5f['dones'][local_episode, step_idx]

        # Convert to tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0  # HWC → CHW, scale to [0,1]
        action = torch.tensor(action, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.bool)

        # Apply transform if provided
        if self.transform:
            image = self.transform(image)

        return image, action, reward, done

    def get_episode(self, episode_idx):
        """
        Get a complete episode.

        Args:
            episode_idx (int): Episode index (0 to total_episodes-1)

        Returns:
            dict: {
                'images': Tensor (steps, C, H, W),
                'actions': Tensor (steps,),
                'rewards': Tensor (steps,),
                'dones': Tensor (steps,)
            }
        """
        chunk_idx, local_episode = self._get_chunk_for_episode(episode_idx)
        chunk_path = self.chunk_paths[chunk_idx]

        if self.use_s3:
            with self.fs.open(chunk_path, 'rb') as f:
                with h5py.File(f, 'r') as h5f:
                    images = h5f['images'][local_episode]
                    actions = h5f['actions'][local_episode]
                    rewards = h5f['rewards'][local_episode]
                    dones = h5f['dones'][local_episode]
        else:
            with h5py.File(chunk_path, 'r') as h5f:
                images = h5f['images'][local_episode]
                actions = h5f['actions'][local_episode]
                rewards = h5f['rewards'][local_episode]
                dones = h5f['dones'][local_episode]

        # Convert to tensors
        images = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0  # NHWC → NCHW
        actions = torch.from_numpy(actions).float()
        rewards = torch.from_numpy(rewards).float()
        dones = torch.from_numpy(dones)

        return {
            'images': images,
            'actions': actions,
            'rewards': rewards,
            'dones': dones
        }


def create_chunked_dataset(
    s3_bucket: str,
    s3_chunk_prefix: str,
    use_s3: bool = True,
    transform: Optional[callable] = None
) -> ChunkedHDF5Dataset:
    """
    Convenience function to create a ChunkedHDF5Dataset from S3.

    Args:
        s3_bucket (str): S3 bucket name
        s3_chunk_prefix (str): S3 prefix containing chunk files
        use_s3 (bool): If True, read from S3. If False, assume local paths
        transform (callable, optional): Transform for images

    Returns:
        ChunkedHDF5Dataset: Dataset ready for use
    """
    if use_s3:
        fs = s3fs.S3FileSystem()
        s3_path = f"{s3_bucket}/{s3_chunk_prefix}"

        # List all chunk files
        all_files = fs.ls(s3_path)
        chunk_files = [f for f in all_files if f.endswith('.h5')]
        chunk_files.sort()

        if not chunk_files:
            raise ValueError(f"No chunk files found at s3://{s3_path}")

        # Paths for S3 dataset
        chunk_paths = chunk_files

    else:
        # Local paths
        import glob
        chunk_files = glob.glob(f"{s3_chunk_prefix}/*.h5")
        chunk_files.sort()

        if not chunk_files:
            raise ValueError(f"No chunk files found at {s3_chunk_prefix}")

        chunk_paths = chunk_files

    return ChunkedHDF5Dataset(
        chunk_paths=chunk_paths,
        use_s3=use_s3,
        transform=transform
    )
