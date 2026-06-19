import os
import h5py
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def precompute_latents(
    vae,
    input_h5_path='car_racing_data.h5',
    output_h5_path='car_racing_latents.h5',
    batch_size=32
):
    """
    Precompute latents for all episodes and save to a new HDF5 file.
    
    Args:
        vae: Trained VAE model
        input_h5_path: Path to original dataset with images
        output_h5_path: Path where latents will be saved
        batch_size: Number of images to encode at once
    """

    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_h5_path), exist_ok=True)
    
    # Setup
    vae.eval()
    vae = vae.to(DEVICE)
    transform = transforms.ToTensor()
    
    # Open input file to get dimensions
    with h5py.File(input_h5_path, 'r') as h5f_in:
        num_episodes = h5f_in['images'].shape[0]
        max_steps    = h5f_in['images'].shape[1]
        action_dim   = 1
        
        print(f"Found {num_episodes} episodes with up to {max_steps} steps each")
        print(f"Action dimension: {action_dim}")
        
        # Get latent dimension by running a single forward pass
        sample_img = h5f_in['images'][0, 0]
        sample_tensor = transform(sample_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            mu, logvar = vae.encode(sample_tensor)
            z_dim = mu.shape[1]
        print(f"Latent dimension: {z_dim}")
    
    # Create output HDF5 file
    with h5py.File(output_h5_path, 'w') as h5f_out:
        # Create datasets with known shapes
        # We'll use variable-length datasets since episodes may have different lengths
        z_dtype = np.float32
        action_dtype = np.float32
        reward_dtype = np.float32
        done_dtype = np.float32
        
        # Create datasets to store variable-length sequences
        z_dataset = h5f_out.create_dataset(
            'z', 
            (num_episodes, max_steps, z_dim),
            dtype=z_dtype,
            chunks=True,
            compression='gzip',
            compression_opts=4
        )
        
        z_next_dataset = h5f_out.create_dataset(
            'z_next',
            (num_episodes, max_steps, z_dim),
            dtype=z_dtype,
            chunks=True,
            compression='gzip',
            compression_opts=4
        )
        
        actions_dataset = h5f_out.create_dataset(
            'actions',
            (num_episodes, max_steps, action_dim),
            dtype=action_dtype,
            chunks=True,
            compression='gzip',
            compression_opts=4
        )
        
        rewards_dataset = h5f_out.create_dataset(
            'rewards',
            (num_episodes, max_steps),
            dtype=reward_dtype,
            chunks=True,
            compression='gzip',
            compression_opts=4
        )
        
        dones_dataset = h5f_out.create_dataset(
            'dones',
            (num_episodes, max_steps),
            dtype=done_dtype,
            chunks=True,
            compression='gzip',
            compression_opts=4
        )
        
        # Store metadata
        h5f_out.attrs['num_episodes'] = num_episodes
        h5f_out.attrs['max_steps'] = max_steps
        h5f_out.attrs['z_dim'] = z_dim
        h5f_out.attrs['action_dim'] = action_dim
    
    # Process episodes one by one
    with h5py.File(input_h5_path, 'r') as h5f_in:
        with h5py.File(output_h5_path, 'a') as h5f_out:
            
            for episode_idx in tqdm(range(num_episodes), desc="Encoding episodes"):
                # Load episode data
                images = h5f_in['images'][episode_idx]
                actions = h5f_in['actions'][episode_idx]
                rewards = h5f_in['rewards'][episode_idx]
                dones = h5f_in['dones'][episode_idx]
                
                # Find actual episode length (where done=1 or end of sequence)
                # Assuming done=1 at the end of each episode
                episode_length = len(images)
                # If there's a done flag in the middle, you might truncate earlier
                # For simplicity, we'll use the full length
                
                # Encode images in batches to avoid memory issues
                all_z = []
                
                for start_idx in range(0, episode_length, batch_size):
                    end_idx = min(start_idx + batch_size, episode_length)
                    batch_images = images[start_idx:end_idx]
                    
                    # Convert batch of images to tensors
                    batch_tensors = torch.stack([
                        transform(img) for img in batch_images
                    ]).to(DEVICE)
                    
                    # Encode
                    with torch.no_grad():
                        mu, logvar = vae.encode(batch_tensors)
                        z_batch = vae.reparameterize(mu, logvar)
                    
                    all_z.append(z_batch.cpu().numpy())
                
                # Concatenate all batches
                z_full = np.concatenate(all_z, axis=0)  # Shape: (episode_length, z_dim)
                
                # Create shifted versions (z_t and z_{t+1})
                z = z_full[:-1]  # All except last
                z_next = z_full[1:]  # All except first
                # actions_shifted = actions[:-1]
                actions_shifted = np.expand_dims(actions, 1)[:-1]
                rewards_shifted = rewards[:-1]
                dones_shifted = dones[:-1]
                
                # Store in datasets (pad with zeros for unused steps)
                seq_len = len(z)
                h5f_out['z'][episode_idx, :seq_len]       = z
                h5f_out['z_next'][episode_idx, :seq_len]  = z_next
                h5f_out['actions'][episode_idx, :seq_len] = actions_shifted
                h5f_out['rewards'][episode_idx, :seq_len] = rewards_shifted
                h5f_out['dones'][episode_idx, :seq_len]   = dones_shifted
                
                # Store actual sequence length as an attribute for this episode
                h5f_out[f'episode_{episode_idx}_length'] = seq_len
    
            print(f"Precomputation complete! Saved to {output_h5_path}")