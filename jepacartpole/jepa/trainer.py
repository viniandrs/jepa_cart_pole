"""
JEPA predictor training module.

Trains the latent predictor on the precomputed latent dataset stored locally.
"""

from pathlib import Path
from typing import Optional

import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from ..utils import save_checkpoint, get_checkpoint_dir, get_data_dir, set_seed
from .dataset import JEPADataset, split_episodes


def sample_latent(mu, logvar):
    """Sample z = mu + sigma * eps (reparameterization, as in World Models)."""
    return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)


@torch.no_grad()
def evaluate_jepa(model, dataloader, device):
    """
    Evaluate the predictor on the latent means (deterministic).

    Returns:
        tuple: (model_mse, copy_last_mse), where copy_last_mse is the error of
               the baseline that predicts z_{t+1} = z_t
    """
    model.eval()
    total, total_baseline, count = 0.0, 0.0, 0
    for mu_ctx, _, action, mu_next, _ in dataloader:
        mu_ctx, action, mu_next = mu_ctx.to(device), action.to(device), mu_next.to(device)
        pred = model(mu_ctx, action)
        total += F.mse_loss(pred, mu_next, reduction='sum').item()
        total_baseline += F.mse_loss(mu_ctx[:, -1], mu_next, reduction='sum').item()
        count += mu_next.numel()
    model.train()
    return total / count, total_baseline / count


def train_jepa(
    model: nn.Module,
    config,
    h5_path: Optional[str] = None,
    epochs: Optional[int] = None,
    checkpoint_dir: Optional[str] = None,
    num_workers: int = 0,
    device: Optional[torch.device] = None,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Train the JEPA predictor on the precomputed latent dataset.

    Episodes are split into train/validation sets (config.val_fraction).
    If config.sample_latents, training inputs and targets are sampled from
    N(mu, sigma^2) on every batch; validation always uses the latent means. Saves a checkpoint to
    jepa_latest.pth after every epoch and the final model to jepa.pth.

    Args:
        model: JEPAPredictor instance
        config: ConfigJEPA with hyperparameters
        h5_path: Path to the latent dataset (default: data/jepa/vae_latents.h5)
        epochs: Number of epochs to train (default: config.epochs)
        checkpoint_dir: Directory for checkpoints (default: project checkpoints/)
        num_workers: DataLoader num_workers (data is in RAM, 0 is enough)
        device: torch.device (defaults to model's current device)
        seed: Random seed for reproducibility (also fixes the episode split)
        verbose: Print progress information

    Returns:
        dict with keys:
            - epoch_train_losses: list of avg train MSE per epoch
            - epoch_val_losses: list of val MSE per epoch (latent means)
            - baseline_val_loss: val MSE of the copy-last baseline
            - best_val_loss: float, best epoch val loss achieved
            - total_batches: int
            - train_episodes, val_episodes: episode indices of the split
    """
    set_seed(seed)

    if epochs is None:
        epochs = config.epochs

    if device is None:
        device = next(model.parameters()).device

    if h5_path is None:
        h5_path = get_data_dir('jepa') / 'vae_latents.h5'

    checkpoint_dir = get_checkpoint_dir() if checkpoint_dir is None else Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, 'r') as h5f:
        num_episodes = h5f['mu'].shape[0]
    train_episodes, val_episodes = split_episodes(num_episodes, config.val_fraction, seed)

    train_dataset = JEPADataset(h5_path, config.context_len, train_episodes)
    val_dataset = JEPADataset(h5_path, config.context_len, val_episodes)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=1024, num_workers=num_workers)

    if verbose:
        print(f"\n{'='*80}")
        print(f"JEPA PREDICTOR TRAINING")
        print(f"{'='*80}")
        print(f"Dataset: {h5_path}")
        print(f"Train: {len(train_episodes)} episodes, {len(train_dataset):,} transitions")
        print(f"Val:   {len(val_episodes)} episodes, {len(val_dataset):,} transitions")
        print(f"Context length: {config.context_len}")
        print(f"Sample latents: {config.sample_latents}")
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

    _, baseline_val_loss = evaluate_jepa(model, val_loader, device)
    if verbose:
        print(f"Copy-last baseline val MSE: {baseline_val_loss:.6f}")

    epoch_train_losses = []
    epoch_val_losses = []

    best_val_loss = float('inf')
    total_batches = 0

    for epoch in range(epochs):
        epoch_total_loss = 0.0
        epoch_batches = 0

        batch_progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            leave=False
        ) if verbose else train_loader

        for mu_ctx, logvar_ctx, action, mu_next, logvar_next in batch_progress:
            mu_ctx, logvar_ctx = mu_ctx.to(device), logvar_ctx.to(device)
            mu_next, logvar_next = mu_next.to(device), logvar_next.to(device)
            action = action.to(device)

            # Sample latents for context and target
            if config.sample_latents:
                z_ctx = sample_latent(mu_ctx, logvar_ctx)
                z_next = sample_latent(mu_next, logvar_next)
            else:
                z_ctx, z_next = mu_ctx, mu_next

            # Forward pass
            z_pred = model(z_ctx, action)
            loss = F.mse_loss(z_pred, z_next)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_total_loss += loss.item()
            epoch_batches += 1

            if verbose:
                batch_progress.set_postfix({'loss': f"{loss.item():.4f}"})

        total_batches += epoch_batches

        epoch_train_losses.append(epoch_total_loss / epoch_batches)
        val_loss, _ = evaluate_jepa(model, val_loader, device)
        epoch_val_losses.append(val_loss)

        best_val_loss = min(best_val_loss, val_loss)

        save_checkpoint(
            model,
            checkpoint_dir / 'jepa_latest.pth',
            epoch=epoch,
            optimizer_state_dict=optimizer.state_dict(),
            config=config,
            best_val_loss=best_val_loss,
            baseline_val_loss=baseline_val_loss,
            epoch_train_losses=epoch_train_losses,
            epoch_val_losses=epoch_val_losses
        )

        if verbose:
            print(f"Epoch {epoch + 1}/{epochs}: "
                  f"train={epoch_train_losses[-1]:.6f}  "
                  f"val={val_loss:.6f}")

    # Save final checkpoint
    save_checkpoint(
        model,
        checkpoint_dir / 'jepa.pth',
        epoch=epochs - 1,
        optimizer_state_dict=optimizer.state_dict(),
        config=config,
        best_val_loss=best_val_loss,
        baseline_val_loss=baseline_val_loss,
        epoch_train_losses=epoch_train_losses,
        epoch_val_losses=epoch_val_losses,
        val_episodes=val_episodes
    )

    if verbose:
        print(f"\n{'='*80}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*80}")
        print(f"Final model saved to: {checkpoint_dir / 'jepa.pth'}")
        print(f"Total batches: {total_batches}")
        print(f"Best val loss: {best_val_loss:.6f} (copy-last baseline: {baseline_val_loss:.6f})")
        print(f"{'='*80}\n")

    return {
        'epoch_train_losses': epoch_train_losses,
        'epoch_val_losses': epoch_val_losses,
        'baseline_val_loss': baseline_val_loss,
        'best_val_loss': best_val_loss,
        'total_batches': total_batches,
        'train_episodes': train_episodes,
        'val_episodes': val_episodes,
    }
