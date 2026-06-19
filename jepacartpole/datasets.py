import h5py
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, List
from torchvision import transforms
import numpy as np

class CartPoleVAEDataset(Dataset):
    """
    PyTorch Dataset for CartPole data stored in HDF5 format.
    
    This dataset provides on-demand access to observations, actions, rewards,
    and done flags stored in the consolidated HDF5 file.
    """
    
    def __init__(self, h5_path='../data/vae/vae_data.h5', transform=None):
        """
        Initialize the dataset.
        
        Args:
            h5_path (str): Path to the HDF5 file containing the dataset
            transform (callable, optional): Transform to apply to the observations
        """
        self.h5_path = h5_path
        
        self.transform = transform if transform is not None else transforms.ToTensor()
        
        # Open the HDF5 file to retrieve dataset dimensions
        with h5py.File(self.h5_path, 'r') as h5f:
            self.num_episodes, self.max_steps = h5f['images'].shape[:2]
            self.total_frames = self.num_episodes * self.max_steps
    
        self.h5_file = None  # Will be opened on first access
    
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
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
        
        # Calculate episode and step from the flat index
        episode = idx // self.max_steps
        step = idx % self.max_steps
        
        # Access the datasets directly using episode and step indices
        image = self.h5_file['images'][episode, step]       
        action = self.h5_file['actions'][episode, step]      
        reward = self.h5_file['rewards'][episode, step]     
        done = self.h5_file['dones'][episode, step]          
        
        # Apply the transform (e.g., ToTensor)
        image = self.transform(image)  # Converts to [C, H, W] and scales to [0, 1]
        
        # Convert action, reward, and done to tensors
        action = torch.tensor(action, dtype=torch.float32)  
        reward = torch.tensor(reward, dtype=torch.float32)  
        done = torch.tensor(done, dtype=torch.float32)     
        
        return image, action, reward, done
    
    def __del__(self):
        """Ensure the HDF5 file is closed when the dataset is deleted."""
        if self.h5_file is not None:
            self.h5_file.close()

class JEPADataset(Dataset):
    """
    Dataset for JEPA training that loads precomputed latents.
    
    Returns transitions (z_t, action_t, z_{t+1}) for each timestep.
    """
    
    def __init__(
        self, 
        h5_path='../data/jepa/jepa.h5'
    ):
        """
        Args:
            h5_path: Path to precomputed latents HDF5 file
        """
        self.h5_path = h5_path
        
        # Open HDF5 to get metadata
        with h5py.File(self.h5_path, 'r') as h5f:
            self.num_episodes = h5f.attrs['num_episodes']
            self.max_steps = h5f.attrs['max_steps']
            self.z_dim = h5f.attrs['z_dim']
            self.action_dim = h5f.attrs['action_dim']

            self.transition_indices = []
            self.episode_lengths = []
            for ep_idx in range(self.num_episodes):
                length = h5f[f'episode_{ep_idx}_length'][()]
                self.episode_lengths.append(length)
                
                # Single transitions: each step is a sample
                for step_idx in range(length):
                    self.transition_indices.append((ep_idx, step_idx))
        
        self.h5_file = None
        
        print(f"Initialized JEPADataset with:")
        print(f"  - {len(self.transition_indices)} total transitions")
        print(f"  - Latent dimension: {self.z_dim}")
        print(f"  - Action dimension: {self.action_dim}")

    def __len__(self):
        return self.num_episodes
    
    def __getitem__(self, idx):
        """
        Returns:
            tuple (z_t, action_t, z_{t+1}, reward_t, done_t)
        """

        ep_idx, start_idx = self.transition_indices[idx]

        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')

        length = self.episode_lengths[ep_idx]
        z_ep = self.h5_file['z'][ep_idx, :length]
        z_next_ep = self.h5_file['z_next'][ep_idx, :length]
        actions_ep = self.h5_file['actions'][ep_idx, :length]
        rewards_ep = self.h5_file['rewards'][ep_idx, :length]
        dones_ep = self.h5_file['dones'][ep_idx, :length]
        
        # Convert to tensors
        z_ep = torch.from_numpy(z_ep).float()
        z_next_ep = torch.from_numpy(z_next_ep).float()
        actions_ep = torch.from_numpy(actions_ep).float()
        rewards_ep = torch.from_numpy(rewards_ep).float()
        dones_ep = torch.from_numpy(dones_ep).float()

        # Single transition
        z_t = z_ep[start_idx]
        action_t = actions_ep[start_idx]
        z_next = z_next_ep[start_idx]
        reward_t = rewards_ep[start_idx]
        done_t = dones_ep[start_idx]
        
        return z_t, action_t, z_next, reward_t, done_t
    
    def __del__(self):
        if self.h5_file is not None:
            self.h5_file.close()