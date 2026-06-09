import h5py
import torch
from torch.utils.data import Dataset
from torchvision import transforms

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