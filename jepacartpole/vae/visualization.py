"""
Visualization utilities for VAE training and evaluation.

Functions for plotting training curves, displaying reconstructions,
and sampling images from S3 chunks.
"""

import os
import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from ..storage import S3Manager
from .dataset import CartPoleVAEDataset


def plot_training_curves(
    loss_history: dict,
    save_path: Optional[str] = None,
    figsize: tuple = (15, 5)
) -> plt.Figure:
    """
    Plot training loss curves from train_vae() output.

    Creates a 1x3 subplot showing:
    - Total loss per epoch
    - Reconstruction loss per epoch
    - KL divergence loss per epoch

    Args:
        loss_history: Dict returned by train_vae() with keys:
                      'epoch_losses', 'epoch_recon_losses', 'epoch_kl_losses'
        save_path: Optional path to save the figure
        figsize: Figure size (width, height)

    Returns:
        matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    epochs = range(1, len(loss_history['epoch_losses']) + 1)

    # Total loss
    axes[0].plot(epochs, loss_history['epoch_losses'], 'o-', linewidth=2, markersize=6)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Total Loss', fontsize=12)
    axes[0].set_title('Total Loss per Epoch', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Reconstruction loss
    axes[1].plot(epochs, loss_history['epoch_recon_losses'], 'o-',
                 linewidth=2, markersize=6, color='orange')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Reconstruction Loss (MSE)', fontsize=12)
    axes[1].set_title('Reconstruction Loss per Epoch', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # KL divergence
    axes[2].plot(epochs, loss_history['epoch_kl_losses'], 'o-',
                 linewidth=2, markersize=6, color='green')
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('KL Divergence', fontsize=12)
    axes[2].set_title('KL Divergence per Epoch', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")

    return fig


def plot_reconstructions(
    model: nn.Module,
    images: torch.Tensor,
    n: int = 5,
    save_path: Optional[str] = None,
    figsize: tuple = (15, 6)
) -> plt.Figure:
    """
    Display original images alongside their VAE reconstructions.

    Args:
        model: Trained VAE model
        images: Batch of images (shape: [batch, C, H, W]) already on the correct device
        n: Number of images to display (must be <= batch size)
        save_path: Optional path to save the figure
        figsize: Figure size (width, height)

    Returns:
        matplotlib.figure.Figure
    """
    model.eval()

    with torch.no_grad():
        reconstructed, _, _ = model(images[:n])

    # Move to CPU and convert to numpy
    images_np = images[:n].cpu().permute(0, 2, 3, 1).numpy()  # [N, H, W, C]
    recon_np = reconstructed.cpu().permute(0, 2, 3, 1).numpy()  # [N, H, W, C]

    # Create figure
    fig, axes = plt.subplots(2, n, figsize=figsize)

    for i in range(n):
        # Original
        axes[0, i].imshow(images_np[i])
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=12, fontweight='bold', loc='left')

        # Reconstructed
        axes[1, i].imshow(recon_np[i])
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed', fontsize=12, fontweight='bold', loc='left')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Reconstructions saved to: {save_path}")

    return fig


def get_sample_images(
    s3_bucket: str,
    s3_prefix: str = 'chunks/dataset',
    n: int = 8,
    local_temp_dir: str = '/tmp/vae_viz',
    seed: int = 42
) -> torch.Tensor:
    """
    Download a single random chunk, sample n random images, return as tensor, clean up.

    Utility for notebooks to get sample images for visualization without keeping
    a chunk around permanently.

    Args:
        s3_bucket: S3 bucket name
        s3_prefix: S3 prefix where chunks are stored
        n: Number of random images to sample
        local_temp_dir: Local directory for temporary chunk download
        seed: Random seed for reproducibility

    Returns:
        torch.Tensor of shape [n, C, H, W] with values in [0, 1]
    """
    local_temp_dir = Path(local_temp_dir)
    local_temp_dir.mkdir(parents=True, exist_ok=True)

    # List chunks
    s3_manager = S3Manager(s3_bucket, verbose=False)
    chunk_keys = s3_manager.list_files(s3_prefix)

    if not chunk_keys:
        raise ValueError(f"No chunk files found in s3://{s3_bucket}/{s3_prefix}")

    # Pick a random chunk
    random.seed(seed)
    chunk_key = random.choice(chunk_keys)
    local_chunk_path = local_temp_dir / Path(chunk_key).name

    try:
        # Download chunk
        print(f"Downloading chunk: {Path(chunk_key).name}")
        success = s3_manager.download_file(chunk_key, str(local_chunk_path))
        if not success:
            raise RuntimeError(f"Failed to download {chunk_key}")

        # Load dataset and sample
        dataset = CartPoleVAEDataset(h5_path=str(local_chunk_path))

        # Sample n random indices
        indices = random.sample(range(len(dataset)), min(n, len(dataset)))

        # Collect images
        images = []
        for idx in indices:
            image, _, _, _ = dataset[idx]
            images.append(image)

        images = torch.stack(images)  # [n, C, H, W]

        # Clean up
        dataset.close()
        del dataset

        print(f"Sampled {len(images)} images from {Path(chunk_key).name}")

        return images

    finally:
        # Always delete the local chunk file
        if local_chunk_path.exists():
            os.remove(local_chunk_path)
            print(f"Deleted local chunk: {local_chunk_path.name}")
