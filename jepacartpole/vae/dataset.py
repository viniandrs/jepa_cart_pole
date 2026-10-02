import h5py
import torch
from torch.utils.data import Dataset

from ..utils import get_data_dir

class CartPoleVAEDataset(Dataset):
    """
    PyTorch Dataset for CartPole data stored in HDF5 format.

    By default all images are loaded into RAM once (the full 64x64 dataset is
    ~3 GB), as in World Models. With in_memory=False, frames are read lazily
    from the HDF5 file on each access.
    """

    def __init__(self, h5_path=None, transform=None, in_memory=True):
        """
        Initialize the dataset.

        Args:
            h5_path (str, optional): Path to the HDF5 file containing the dataset.
                                     Defaults to data/vae/vae_data.h5
            transform (callable, optional): Transform applied to the image tensor
                                            (C, H, W) in [0, 1]
            in_memory (bool): If True, load the whole dataset into RAM
        """
        if h5_path is None:
            h5_path = get_data_dir('vae') / 'vae_data.h5'
        self.h5_path = str(h5_path)
        self.transform = transform
        self.in_memory = in_memory

        with h5py.File(self.h5_path, 'r') as h5f:
            self.num_episodes, self.max_steps = h5f['images'].shape[:2]
            self.total_frames = self.num_episodes * self.max_steps

            if in_memory:
                self.data = {key: h5f[key][:] for key in ('images', 'actions', 'rewards', 'dones')}

        self.h5_file = None  # Opened on first access when not in memory

    def __len__(self):
        """Return the total number of frames in the dataset."""
        return self.total_frames

    def __getitem__(self, idx):
        """
        Retrieve a single frame and associated data.

        Args:
            idx (int): Index of the frame to retrieve

        Returns:
            tuple: (image, action, reward, done) tensors
        """
        # Calculate episode and step from the flat index
        episode = idx // self.max_steps
        step = idx % self.max_steps

        if self.in_memory:
            source = self.data
        else:
            if self.h5_file is None:
                self.h5_file = h5py.File(self.h5_path, 'r')
            source = self.h5_file

        image = source['images'][episode, step]
        action = source['actions'][episode, step]
        reward = source['rewards'][episode, step]
        done = source['dones'][episode, step]

        # HWC uint8 -> CHW float in [0, 1]
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        if self.transform is not None:
            image = self.transform(image)

        action = torch.tensor(action, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)

        return image, action, reward, done

    def close(self):
        """Explicitly close the HDF5 file handle."""
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None

    def __del__(self):
        """Ensure the HDF5 file is closed when the dataset is deleted."""
        self.close()
