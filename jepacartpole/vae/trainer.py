"""
VAE training module with chunked S3 data support.

This module implements chunk-streaming training: download one chunk at a time,
train on it, delete it, repeat. This keeps disk usage minimal (~15GB peak)
while training on large datasets stored in S3.
"""

import os
import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from ..storage import S3Manager
from ..utils import save_checkpoint, get_checkpoint_dir
from .dataset import CartPoleVAEDataset


def train_vae(
    model: nn.Module,
    config,
    s3_bucket: str,
    s3_prefix: str = 'chunks/dataset',
    local_temp_dir: str = '/tmp/vae_training',
    checkpoint_dir: Optional[str] = None,
    num_workers: int = 2,
    device: Optional[torch.device] = None,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Train VAE model using chunked S3 data with minimal local disk usage.

    Strategy: Download one chunk at a time (~15GB), train on it, delete it,
    repeat. Chunk order is shuffled each epoch for diversity.

    Args:
        model: VAE model instance
        config: ConfigVAE with hyperparameters (batch_size, epochs, learning_rate, etc.)
        s3_bucket: S3 bucket name
        s3_prefix: S3 prefix where chunks are stored (default: 'chunks/dataset')
        local_temp_dir: Local directory for temporary chunk downloads
        checkpoint_dir: Directory for checkpoints (default: project checkpoints/)
        num_workers: DataLoader num_workers
        device: torch.device (defaults to model's current device)
        seed: Random seed for reproducibility
        verbose: Print progress information

    Returns:
        dict with keys:
            - epoch_losses: list of avg total loss per epoch
            - epoch_recon_losses: list of avg recon loss per epoch
            - epoch_kl_losses: list of avg KL loss per epoch
            - chunk_losses: list of avg total loss per chunk (all epochs)
            - best_loss: float, best total loss achieved
            - total_batches: int
            - total_chunks_processed: int
    """
    # Setup
    if device is None:
        device = next(model.parameters()).device

    if checkpoint_dir is None:
        checkpoint_dir = get_checkpoint_dir()
    else:
        checkpoint_dir = Path(checkpoint_dir)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    local_temp_dir = Path(local_temp_dir)
    local_temp_dir.mkdir(parents=True, exist_ok=True)

    # Initialize S3 manager and list chunks
    s3_manager = S3Manager(s3_bucket, verbose=verbose)
    chunk_keys = s3_manager.list_files(s3_prefix)

    if not chunk_keys:
        raise ValueError(f"No chunk files found in s3://{s3_bucket}/{s3_prefix}")

    chunk_keys = sorted(chunk_keys)  # Ensure consistent base order
    num_chunks = len(chunk_keys)

    if verbose:
        print(f"\n{'='*80}")
        print(f"VAE TRAINING - CHUNK STREAMING MODE")
        print(f"{'='*80}")
        print(f"S3 location: s3://{s3_bucket}/{s3_prefix}")
        print(f"Chunks: {num_chunks}")
        print(f"Epochs: {config.epochs}")
        print(f"Batch size: {config.batch_size}")
        print(f"Learning rate: {config.learning_rate}")
        print(f"Weight decay: {config.weight_decay}")
        print(f"Local temp dir: {local_temp_dir}")
        print(f"Device: {device}")
        print(f"{'='*80}\n")

    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Training state
    model.to(device)
    model.train()

    # Loss tracking
    epoch_losses = []
    epoch_recon_losses = []
    epoch_kl_losses = []
    chunk_losses = []  # Flat list of all chunk losses

    best_loss = float('inf')
    total_batches = 0
    total_chunks_processed = 0

    # Training loop
    for epoch in range(config.epochs):
        epoch_total_loss = 0.0
        epoch_total_recon = 0.0
        epoch_total_kl = 0.0
        epoch_batches = 0

        # Shuffle chunk order for this epoch (deterministic based on seed + epoch)
        epoch_rng = random.Random(seed + epoch)
        shuffled_chunks = chunk_keys.copy()
        epoch_rng.shuffle(shuffled_chunks)

        if verbose:
            print(f"\n{'─'*80}")
            print(f"EPOCH {epoch + 1}/{config.epochs}")
            print(f"{'─'*80}")

        # Process each chunk
        chunk_progress = tqdm(shuffled_chunks, desc=f"Epoch {epoch + 1} chunks") if verbose else shuffled_chunks

        for chunk_idx, chunk_key in enumerate(chunk_progress):
            local_chunk_path = local_temp_dir / Path(chunk_key).name

            try:
                # Download chunk
                if verbose and not isinstance(chunk_progress, tqdm):
                    print(f"  Downloading chunk {chunk_idx + 1}/{num_chunks}: {Path(chunk_key).name}")

                success = s3_manager.download_file(chunk_key, str(local_chunk_path))
                if not success:
                    raise RuntimeError(f"Failed to download {chunk_key}")

                # Create dataset and dataloader
                dataset = CartPoleVAEDataset(h5_path=str(local_chunk_path))
                dataloader = DataLoader(
                    dataset,
                    batch_size=config.batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=True if device.type == 'cuda' else False
                )

                # Train on this chunk
                chunk_total_loss = 0.0
                chunk_total_recon = 0.0
                chunk_total_kl = 0.0
                chunk_batches = 0

                batch_progress = tqdm(
                    dataloader,
                    desc=f"  Chunk {chunk_idx + 1}/{num_chunks}",
                    leave=False
                ) if verbose else dataloader

                for images, actions, rewards, dones in batch_progress:
                    images = images.to(device)

                    # Forward pass
                    recon, mu, logvar = model(images)
                    loss, recon_loss, kl_loss = model.loss(recon, images, mu, logvar)

                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Track losses
                    chunk_total_loss += loss.item()
                    chunk_total_recon += recon_loss.item()
                    chunk_total_kl += kl_loss.item()
                    chunk_batches += 1

                    if verbose and isinstance(batch_progress, tqdm):
                        batch_progress.set_postfix({
                            'loss': f"{loss.item():.2f}",
                            'recon': f"{recon_loss.item():.2f}",
                            'kl': f"{kl_loss.item():.2f}"
                        })

                # Aggregate chunk losses
                avg_chunk_loss = chunk_total_loss / chunk_batches
                avg_chunk_recon = chunk_total_recon / chunk_batches
                avg_chunk_kl = chunk_total_kl / chunk_batches

                chunk_losses.append(avg_chunk_loss)

                epoch_total_loss += chunk_total_loss
                epoch_total_recon += chunk_total_recon
                epoch_total_kl += chunk_total_kl
                epoch_batches += chunk_batches
                total_batches += chunk_batches
                total_chunks_processed += 1

                # Close dataset and cleanup
                dataset.close()
                del dataset
                del dataloader

                # Save checkpoint after each chunk
                if avg_chunk_loss < best_loss:
                    best_loss = avg_chunk_loss

                save_checkpoint(
                    model,
                    checkpoint_dir / 'vae_latest.pth',
                    epoch=epoch,
                    chunk_idx=chunk_idx,
                    optimizer_state_dict=optimizer.state_dict(),
                    config=config,
                    best_loss=best_loss,
                    chunk_losses=chunk_losses,
                    epoch_losses=epoch_losses
                )

                if verbose and isinstance(chunk_progress, tqdm):
                    chunk_progress.set_postfix({
                        'chunk_loss': f"{avg_chunk_loss:.2f}",
                        'best': f"{best_loss:.2f}"
                    })

            finally:
                # Always delete the local chunk file (even on error)
                if local_chunk_path.exists():
                    os.remove(local_chunk_path)
                    if verbose and not isinstance(chunk_progress, tqdm):
                        print(f"  Deleted local chunk: {local_chunk_path.name}")

        # Epoch summary
        avg_epoch_loss = epoch_total_loss / epoch_batches
        avg_epoch_recon = epoch_total_recon / epoch_batches
        avg_epoch_kl = epoch_total_kl / epoch_batches

        epoch_losses.append(avg_epoch_loss)
        epoch_recon_losses.append(avg_epoch_recon)
        epoch_kl_losses.append(avg_epoch_kl)

        if verbose:
            print(f"\nEpoch {epoch + 1} summary:")
            print(f"  Avg loss: {avg_epoch_loss:.4f}")
            print(f"  Avg recon: {avg_epoch_recon:.4f}")
            print(f"  Avg KL: {avg_epoch_kl:.4f}")
            print(f"  Best loss: {best_loss:.4f}")

    # Save final checkpoint
    save_checkpoint(
        model,
        checkpoint_dir / 'vae.pth',
        epoch=config.epochs - 1,
        optimizer_state_dict=optimizer.state_dict(),
        config=config,
        best_loss=best_loss,
        epoch_losses=epoch_losses,
        epoch_recon_losses=epoch_recon_losses,
        epoch_kl_losses=epoch_kl_losses,
        chunk_losses=chunk_losses
    )

    if verbose:
        print(f"\n{'='*80}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*80}")
        print(f"Final model saved to: {checkpoint_dir / 'vae.pth'}")
        print(f"Total batches: {total_batches}")
        print(f"Total chunks processed: {total_chunks_processed}")
        print(f"Best loss: {best_loss:.4f}")
        print(f"{'='*80}\n")

    return {
        'epoch_losses': epoch_losses,
        'epoch_recon_losses': epoch_recon_losses,
        'epoch_kl_losses': epoch_kl_losses,
        'chunk_losses': chunk_losses,
        'best_loss': best_loss,
        'total_batches': total_batches,
        'total_chunks_processed': total_chunks_processed,
    }
