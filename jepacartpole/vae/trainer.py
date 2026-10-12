"""
VAE training module.

Trains the VAE on the consolidated HDF5 dataset stored locally.
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from ..utils import save_checkpoint, get_checkpoint_dir, get_data_dir, set_seed
from .dataset import CartPoleVAEDataset


def train_vae(
    model: nn.Module,
    config,
    h5_path: Optional[str] = None,
    epochs: Optional[int] = None,
    checkpoint_dir: Optional[str] = None,
    num_workers: int = 2,
    device: Optional[torch.device] = None,
    seed: int = 42,
    in_memory: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Train VAE model on the consolidated HDF5 dataset.

    Saves a checkpoint to vae_latest.pth after every epoch and the final
    model to vae.pth.

    Args:
        model: VAE model instance
        config: ConfigVAE with hyperparameters (batch_size, epochs, learning_rate, etc.)
        h5_path: Path to the HDF5 dataset (default: data/vae/vae_data.h5)
        epochs: Number of epochs to train (default: config.epochs). Useful for
                a short validation run without changing the config
        checkpoint_dir: Directory for checkpoints (default: project checkpoints/)
        num_workers: DataLoader num_workers
        device: torch.device (defaults to model's current device)
        seed: Random seed for reproducibility
        in_memory: Load the whole dataset into RAM (see CartPoleVAEDataset)
        verbose: Print progress information

    Returns:
        dict with keys:
            - epoch_losses: list of avg total loss per epoch
            - epoch_recon_losses: list of avg recon loss per epoch
            - epoch_kl_losses: list of avg KL loss per epoch
            - best_loss: float, best epoch total loss achieved
            - total_batches: int
    """
    set_seed(seed)

    if epochs is None:
        epochs = config.epochs

    if device is None:
        device = next(model.parameters()).device

    if h5_path is None:
        h5_path = get_data_dir('vae') / 'vae_data.h5'

    checkpoint_dir = get_checkpoint_dir() if checkpoint_dir is None else Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    dataset = CartPoleVAEDataset(h5_path=h5_path, in_memory=in_memory)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == 'cuda'
    )

    if verbose:
        print(f"\n{'='*80}")
        print(f"VAE TRAINING")
        print(f"{'='*80}")
        print(f"Dataset: {h5_path} ({len(dataset):,} frames)")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {config.batch_size}")
        print(f"Learning rate: {config.learning_rate}")
        print(f"Weight decay: {config.weight_decay}")
        print(f"Device: {device}")
        print(f"{'='*80}\n")

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    model.to(device)
    model.train()

    epoch_losses = []
    epoch_recon_losses = []
    epoch_kl_losses = []

    best_loss = float('inf')
    total_batches = 0

    for epoch in range(epochs):
        epoch_total_loss = 0.0
        epoch_total_recon = 0.0
        epoch_total_kl = 0.0
        epoch_batches = 0

        batch_progress = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            leave=False
        ) if verbose else dataloader

        for images, actions, rewards, dones in batch_progress:
            images = images.to(device, non_blocking=True)

            # Forward pass
            recon, mu, logvar = model(images)
            loss, recon_loss, kl_loss = model.loss(recon, images, mu, logvar)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track losses
            epoch_total_loss += loss.item()
            epoch_total_recon += recon_loss.item()
            epoch_total_kl += kl_loss.item()
            epoch_batches += 1

            if verbose:
                batch_progress.set_postfix({
                    'loss': f"{loss.item():.2f}",
                    'recon': f"{recon_loss.item():.2f}",
                    'kl': f"{kl_loss.item():.2f}"
                })

        total_batches += epoch_batches

        avg_epoch_loss = epoch_total_loss / epoch_batches
        epoch_losses.append(avg_epoch_loss)
        epoch_recon_losses.append(epoch_total_recon / epoch_batches)
        epoch_kl_losses.append(epoch_total_kl / epoch_batches)

        best_loss = min(best_loss, avg_epoch_loss)

        save_checkpoint(
            model,
            checkpoint_dir / 'vae_latest.pth',
            epoch=epoch,
            optimizer_state_dict=optimizer.state_dict(),
            config=config,
            best_loss=best_loss,
            epoch_losses=epoch_losses,
            epoch_recon_losses=epoch_recon_losses,
            epoch_kl_losses=epoch_kl_losses
        )

        if verbose:
            print(f"Epoch {epoch + 1}/{epochs}: "
                  f"loss={avg_epoch_loss:.4f}  "
                  f"recon={epoch_recon_losses[-1]:.4f}  "
                  f"kl={epoch_kl_losses[-1]:.4f}")

    dataset.close()

    # Save final checkpoint
    save_checkpoint(
        model,
        checkpoint_dir / 'vae.pth',
        epoch=epochs - 1,
        optimizer_state_dict=optimizer.state_dict(),
        config=config,
        best_loss=best_loss,
        epoch_losses=epoch_losses,
        epoch_recon_losses=epoch_recon_losses,
        epoch_kl_losses=epoch_kl_losses
    )

    if verbose:
        print(f"\n{'='*80}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*80}")
        print(f"Final model saved to: {checkpoint_dir / 'vae.pth'}")
        print(f"Total batches: {total_batches}")
        print(f"Best loss: {best_loss:.4f}")
        print(f"{'='*80}\n")

    return {
        'epoch_losses': epoch_losses,
        'epoch_recon_losses': epoch_recon_losses,
        'epoch_kl_losses': epoch_kl_losses,
        'best_loss': best_loss,
        'total_batches': total_batches,
    }
