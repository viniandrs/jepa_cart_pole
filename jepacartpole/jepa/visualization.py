"""
Visualization utilities for JEPA predictor training and evaluation.

Functions for plotting training curves, decoding one-step predictions and
evaluating open-loop (imagined) rollouts in latent space.
"""

from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from ..utils import get_data_dir
from .dataset import JEPADataset


def _save(fig, save_path, what):
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"{what} saved to: {save_path}")


def plot_jepa_training_curves(
    loss_history: dict,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 5)
) -> plt.Figure:
    """
    Plot train/validation MSE per epoch from train_jepa() output, with the
    copy-last baseline (z_{t+1} = z_t) as a dashed reference line.

    When training on sampled latents, the train loss includes the variance of
    the sampled target and is not comparable to the validation loss (on mu).

    Args:
        loss_history: Dict returned by train_jepa()
        save_path: Optional path to save the figure
        figsize: Figure size (width, height)

    Returns:
        matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    epochs = range(1, len(loss_history['epoch_train_losses']) + 1)

    ax.plot(epochs, loss_history['epoch_train_losses'], 'o-', linewidth=2,
            markersize=6, label='Train')
    ax.plot(epochs, loss_history['epoch_val_losses'], 'o-', linewidth=2,
            markersize=6, color='orange', label='Validation (μ)')
    ax.axhline(loss_history['baseline_val_loss'], linestyle='--', color='gray',
               label='Copy-last baseline (val)')

    ax.set_yscale('log')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Latent MSE', fontsize=12)
    ax.set_title('JEPA Predictor Loss per Epoch', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    _save(fig, save_path, "Training curves")

    return fig


@torch.no_grad()
def plot_predictions(
    vae: nn.Module,
    predictor: nn.Module,
    dataset: JEPADataset,
    frames_h5_path: Optional[str] = None,
    n: int = 5,
    seed: int = 42,
    save_path: Optional[str] = None,
    figsize: tuple = (15, 9)
) -> plt.Figure:
    """
    Decode one-step predictions for n random transitions.

    Rows: true frame t+1, VAE reconstruction of mu_{t+1} (the best the
    predictor can achieve through the decoder), decoded prediction.

    Args:
        vae: Trained VAE (used only to decode latents)
        predictor: Trained JEPAPredictor
        dataset: JEPADataset (e.g. validation episodes)
        frames_h5_path: Frame dataset (default: data/vae/vae_data.h5)
        n: Number of transitions to display
        seed: Random seed for the transition sample
        save_path: Optional path to save the figure
        figsize: Figure size (width, height)

    Returns:
        matplotlib.figure.Figure
    """
    if frames_h5_path is None:
        frames_h5_path = get_data_dir('vae') / 'vae_data.h5'

    device = next(predictor.parameters()).device
    vae.eval()
    predictor.eval()

    indices = np.random.default_rng(seed).choice(len(dataset), n, replace=False)
    mu_ctx, _, action, mu_next, _ = zip(*(dataset[i] for i in indices))
    mu_ctx, action, mu_next = (torch.stack(x).to(device) for x in (mu_ctx, action, mu_next))

    target_recon = vae.decode(mu_next).cpu().permute(0, 2, 3, 1).numpy()
    pred_recon = vae.decode(predictor(mu_ctx, action)).cpu().permute(0, 2, 3, 1).numpy()

    with h5py.File(frames_h5_path, 'r') as h5f:
        true_frames = []
        for i in indices:
            ep, t = dataset.index[i].tolist()
            true_frames.append(h5f['images'][dataset.episodes[ep], t + 1])

    rows = [('True frame t+1', true_frames),
            ('VAE recon of z_{t+1}', target_recon),
            ('Decoded prediction', pred_recon)]

    fig, axes = plt.subplots(len(rows), n, figsize=figsize)
    for r, (title, images) in enumerate(rows):
        for i in range(n):
            axes[r, i].imshow(images[i])
            axes[r, i].axis('off')
        axes[r, 0].set_title(title, fontsize=12, fontweight='bold', loc='left')

    plt.tight_layout()
    _save(fig, save_path, "Predictions")

    return fig


@torch.no_grad()
def rollout(predictor: nn.Module, mu_ctx: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """
    Open-loop (imagined) rollout: feed predictions back as context.

    Args:
        predictor: Trained JEPAPredictor
        mu_ctx: (B, context_len, z_dim) initial latent context
        actions: (B, horizon) actions taken from the last context step on

    Returns:
        (B, horizon, z_dim) predicted latents z_{t+1}, ..., z_{t+horizon}
    """
    predictor.eval()
    ctx = mu_ctx
    preds = []
    for h in range(actions.shape[1]):
        z_next = predictor(ctx, actions[:, h:h + 1])
        preds.append(z_next)
        ctx = torch.cat([ctx[:, 1:], z_next.unsqueeze(1)], dim=1)
    return torch.stack(preds, dim=1)


def sample_rollout_segments(dataset: JEPADataset, horizon: int, n: int = 500, seed: int = 42):
    """
    Sample n start points (episode, t) whose context and the next `horizon`
    targets contain no environment reset.

    Returns:
        tuple: (mu_ctx (n, k, z), actions (n, horizon), mu_future (n, horizon, z),
                starts (n, 2) array of (episode, t) in dataset episode indexing)
    """
    k = dataset.context_len
    span = k + horizon - 1  # dones[t-k+1 .. t+horizon-1] must all be False
    starts = [(ep, t) for ep, t in dataset.index.tolist()
              if t + horizon < dataset.max_steps
              and not dataset.dones[ep, t - k + 1:t - k + 1 + span].any()]
    if not starts:
        raise ValueError(f"No reset-free segments of length {horizon} in the dataset.")
    rng = np.random.default_rng(seed)
    starts = np.array(starts)[rng.choice(len(starts), min(n, len(starts)), replace=False)]

    mu_ctx = torch.stack([dataset.mu[ep, t - k + 1:t + 1] for ep, t in starts])
    actions = torch.stack([dataset.actions[ep, t:t + horizon] for ep, t in starts])
    mu_future = torch.stack([dataset.mu[ep, t + 1:t + 1 + horizon] for ep, t in starts])
    return mu_ctx, actions, mu_future, starts


def plot_rollout_error(
    predictor: nn.Module,
    dataset: JEPADataset,
    horizon: int = 8,
    n: int = 500,
    seed: int = 42,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 5)
) -> plt.Figure:
    """
    Plot the latent MSE of open-loop rollouts as a function of the horizon,
    against the copy-last baseline (z_{t+h} = z_t).

    Returns:
        matplotlib.figure.Figure
    """
    device = next(predictor.parameters()).device
    mu_ctx, actions, mu_future, _ = sample_rollout_segments(dataset, horizon, n, seed)
    mu_ctx, actions, mu_future = mu_ctx.to(device), actions.to(device), mu_future.to(device)

    preds = rollout(predictor, mu_ctx, actions)
    model_err = (preds - mu_future).pow(2).mean(dim=(0, 2)).cpu().numpy()
    baseline_err = (mu_ctx[:, -1:] - mu_future).pow(2).mean(dim=(0, 2)).cpu().numpy()

    steps = np.arange(1, horizon + 1)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(steps, model_err, 'o-', linewidth=2, markersize=6, label='JEPA rollout')
    ax.plot(steps, baseline_err, 'o--', linewidth=2, markersize=6, color='gray',
            label='Copy-last baseline')
    ax.set_xlabel('Prediction horizon (steps)', fontsize=12)
    ax.set_ylabel('Latent MSE', fontsize=12)
    ax.set_title(f'Open-loop Rollout Error ({len(preds)} segments)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    _save(fig, save_path, "Rollout error")

    return fig


@torch.no_grad()
def plot_rollout_frames(
    vae: nn.Module,
    predictor: nn.Module,
    dataset: JEPADataset,
    frames_h5_path: Optional[str] = None,
    horizon: int = 8,
    seed: int = 0,
    save_path: Optional[str] = None,
    figsize: tuple = (18, 4)
) -> plt.Figure:
    """
    Show one imagined rollout: true frames vs decoded predictions over the
    horizon, for a random reset-free segment.

    Returns:
        matplotlib.figure.Figure
    """
    if frames_h5_path is None:
        frames_h5_path = get_data_dir('vae') / 'vae_data.h5'

    device = next(predictor.parameters()).device
    vae.eval()
    mu_ctx, actions, _, starts = sample_rollout_segments(dataset, horizon, n=1, seed=seed)
    preds = rollout(predictor, mu_ctx.to(device), actions.to(device))[0]
    decoded = vae.decode(preds).cpu().permute(0, 2, 3, 1).numpy()

    ep, t = starts[0]
    with h5py.File(frames_h5_path, 'r') as h5f:
        true_frames = h5f['images'][dataset.episodes[ep], t + 1:t + 1 + horizon]

    fig, axes = plt.subplots(2, horizon, figsize=figsize)
    for h in range(horizon):
        axes[0, h].imshow(true_frames[h])
        axes[1, h].imshow(decoded[h])
        axes[0, h].set_title(f't+{h + 1}\na={int(actions[0, h])}', fontsize=10)
        axes[0, h].axis('off')
        axes[1, h].axis('off')
    axes[0, 0].text(-8, 32, 'True', fontsize=12, fontweight='bold', rotation=90, va='center', ha='right')
    axes[1, 0].text(-8, 32, 'Imagined', fontsize=12, fontweight='bold', rotation=90, va='center', ha='right')

    plt.tight_layout()
    _save(fig, save_path, "Rollout frames")

    return fig
