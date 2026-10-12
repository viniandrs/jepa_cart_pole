"""
Latent encoding of the consolidated frame dataset.

Following World Models (Ha & Schmidhuber, 2018), every frame of the HDF5
dataset is encoded once with a trained encoder and the latent distribution
parameters (mu, logvar) are stored, together with the actions, rewards and
dones, in a new HDF5 file used to train the JEPA predictor.
"""

import os
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

from ..utils import get_data_dir, get_device


def encode_dataset(encoder, h5_path=None, output_path=None, batch_size=500,
                   device=None, overwrite=False, verbose=True):
    """
    Encode every frame of the consolidated dataset into latent parameters.

    The output HDF5 file has the structure:
        mu:      float32 (num_episodes, max_steps, z_dim)
        logvar:  float32 (num_episodes, max_steps, z_dim)
        actions: float32 (num_episodes, max_steps)
        rewards: float32 (num_episodes, max_steps)
        dones:   bool    (num_episodes, max_steps)

    Args:
        encoder (torch.nn.Module): Any model with an encode(x) -> (mu, logvar)
                                   method taking (B, C, H, W) images in [0, 1]
        h5_path (str, optional): Frame dataset. Defaults to data/vae/vae_data.h5
        output_path (str, optional): Output file. Defaults to data/jepa/vae_latents.h5
        batch_size (int): Frames encoded per forward pass
        device (torch.device, optional): Device used for encoding
        overwrite (bool): If False and the output file exists, skip encoding
        verbose (bool): If True, print progress messages

    Returns:
        Path: Path to the latent dataset
    """
    if h5_path is None:
        h5_path = get_data_dir('vae') / 'vae_data.h5'
    if output_path is None:
        output_path = get_data_dir('jepa') / 'vae_latents.h5'
    if device is None:
        device = get_device()
    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        if verbose:
            print(f"✅ Found existing latent dataset: {output_path} (use overwrite=True to re-encode)")
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    encoder.eval()
    encoder.to(device)

    with h5py.File(h5_path, 'r') as h5_in, h5py.File(output_path, 'w') as h5_out:
        num_episodes, max_steps = h5_in['images'].shape[:2]

        # Infer the latent dimension from a single frame
        with torch.no_grad():
            z_dim = encoder.encode(frames_to_tensor(h5_in['images'][0, :1], device))[0].shape[1]

        if verbose:
            print(f"Encoding {num_episodes} episodes × {max_steps} steps "
                  f"({num_episodes * max_steps:,} frames) into z_dim={z_dim}")

        # One HDF5 chunk per episode, as in the frame dataset
        for key in ('mu', 'logvar'):
            h5_out.create_dataset(key, shape=(num_episodes, max_steps, z_dim),
                                  dtype='float32', chunks=(1, max_steps, z_dim))
        for key, dtype in (('actions', 'float32'), ('rewards', 'float32'), ('dones', 'bool')):
            h5_out.create_dataset(key, data=h5_in[key][:].astype(dtype),
                                  chunks=(1, max_steps))

        h5_out.attrs['z_dim'] = z_dim
        h5_out.attrs['source_file'] = str(h5_path)
        for key in ('crop_rows', 'frame_size'):
            if key in h5_in.attrs:
                h5_out.attrs[key] = h5_in.attrs[key]

        iterator = tqdm(range(num_episodes), desc="Encoding episodes") if verbose else range(num_episodes)

        with torch.no_grad():
            for ep in iterator:
                images = h5_in['images'][ep]
                mus, logvars = [], []
                for start in range(0, max_steps, batch_size):
                    mu, logvar = encoder.encode(frames_to_tensor(images[start:start + batch_size], device))
                    mus.append(mu.cpu().numpy())
                    logvars.append(logvar.cpu().numpy())
                h5_out['mu'][ep] = np.concatenate(mus)
                h5_out['logvar'][ep] = np.concatenate(logvars)

    if verbose:
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"Latent dataset saved to: {output_path} ({size_mb:.1f} MB)")

    return output_path


def frames_to_tensor(images, device):
    """Convert uint8 HWC frames to a float CHW tensor in [0, 1]."""
    return torch.from_numpy(images).permute(0, 3, 1, 2).float().div_(255.0).to(device)
