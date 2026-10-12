"""
Visualization utilities for full JEPA training and the latent decoder.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from .dataset import frames_to_device


def _save(fig, save_path, what):
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"{what} saved to: {save_path}")


def plot_full_jepa_training_curves(history: dict, save_path: Optional[str] = None,
                                   figsize: tuple = (20, 4)) -> plt.Figure:
    """
    Five panels from train_full_jepa() output: unweighted loss terms,
    validation skill vs copy-last (target-encoder context, as in evaluation,
    and online-encoder context, as in training), latent std, effective
    dimension and online/target encoder gap. A collapsing encoder shows
    std -> 0 or effective dimension -> 1.
    """
    epochs = np.arange(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 5, figsize=figsize)

    ax = axes[0]
    for key, label in (('train_pred_loss', 'prediction (MSE)'), ('train_var_loss', 'variance'),
                       ('train_cov_loss', 'covariance')):
        ax.plot(epochs, history[key], 'o-', markersize=4, label=label)
    ax.set_yscale('log')
    ax.set_title('Training loss terms (unweighted)', fontweight='bold')
    ax.legend()

    ax = axes[1]
    ax.plot(epochs, history['val_skill'], 'o-', markersize=4, color='tab:green', label='target context')
    if 'val_skill_online' in history:
        ax.plot(epochs, history['val_skill_online'], 's--', markersize=3, color='tab:olive',
                label='online context')
    ax.axhline(0, linestyle='--', color='gray', label='copy-last')
    ax.set_ylim(max(min(history['val_skill']), -1), 1)
    ax.set_title('Val skill vs copy-last', fontweight='bold')
    ax.legend()

    panels = (('val_std', 'Mean latent std (val)', 'tab:purple', False),
              ('val_effective_dim', 'Effective dimension (val)', 'tab:orange', False),
              ('val_encoder_gap', 'Online/target gap (val)', 'tab:red', True))
    for ax, (key, title, color, log) in zip(axes[2:], panels):
        ax.plot(epochs, history[key], 'o-', markersize=4, color=color)
        ax.set_title(title, fontweight='bold')
        if log:
            ax.set_yscale('log')

    for ax in axes:
        ax.set_xlabel('Epoch')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, save_path, "Training curves")
    return fig


def plot_latent_spectrum(z, save_path: Optional[str] = None, figsize: tuple = (12, 4)) -> plt.Figure:
    """Per-dimension std and PCA explained-variance spectrum of latents (..., z_dim)."""
    flat = z.reshape(-1, z.shape[-1])
    eig = np.sort(np.clip(np.linalg.eigvalsh(np.cov(flat, rowvar=False)), 0, None))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].bar(np.arange(flat.shape[1]), flat.std(axis=0))
    axes[0].set_xlabel('Latent dimension')
    axes[0].set_ylabel('std')
    axes[0].set_title('Per-dimension std', fontweight='bold')
    ratio = eig / eig.sum()
    axes[1].bar(np.arange(1, len(eig) + 1), ratio)
    axes[1].plot(np.arange(1, len(eig) + 1), np.cumsum(ratio), 'o-', color='tab:orange',
                 markersize=3, label='cumulative')
    axes[1].set_xlabel('Principal component')
    axes[1].set_ylabel('Explained variance ratio')
    axes[1].set_title('PCA spectrum', fontweight='bold')
    axes[1].legend()
    for ax in axes:
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, save_path, "Latent spectrum")
    return fig


def plot_decoder_training_curves(history: dict, save_path: Optional[str] = None,
                                 figsize: tuple = (11, 4)) -> plt.Figure:
    """Train/val reconstruction loss and val PSNR per epoch from train_latent_decoder()."""
    epochs = np.arange(1, len(history['epoch_train_losses']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].plot(epochs, history['epoch_train_losses'], 'o-', label='Train')
    axes[0].plot(epochs, history['epoch_val_losses'], 'o-', color='orange', label='Validation')
    axes[0].set_ylabel('Sum of squared errors per frame')
    axes[0].set_yscale('log')
    axes[0].legend()
    axes[1].plot(epochs, history['epoch_val_psnr'], 'o-', color='tab:green')
    axes[1].set_ylabel('Val PSNR [dB]')
    for ax in axes:
        ax.set_xlabel('Epoch')
        ax.grid(True, alpha=0.3)
    fig.suptitle('Latent decoder training', fontweight='bold')
    plt.tight_layout()
    _save(fig, save_path, "Decoder curves")
    return fig


@torch.no_grad()
def plot_decoder_reconstructions(encoder, decoder, images, extra=None, save_path: Optional[str] = None):
    """
    Original frames (uint8 (n, H, W, C)) and their reconstructions decode(encode(x)).

    Args:
        extra: optional dict {row label: (encoder, decoder)} of other models to
               add as rows (e.g. the VAE, for comparison)
    """
    device = next(decoder.parameters()).device
    x = frames_to_device(images, device)
    rows = {'Original': images}
    rows['JEPA decoder'] = decoder.decode(encoder.encode(x)).cpu().permute(0, 2, 3, 1).numpy()
    for label, (enc, dec) in (extra or {}).items():
        out = enc.encode(x)
        z = out[0] if isinstance(out, tuple) else out
        rows[label] = dec.decode(z).cpu().permute(0, 2, 3, 1).numpy()

    n = len(images)
    fig, axes = plt.subplots(len(rows), n, figsize=(2.2 * n, 2.3 * len(rows)), squeeze=False)
    for r, (label, imgs) in enumerate(rows.items()):
        for i in range(n):
            axes[r, i].imshow(imgs[i])
            axes[r, i].axis('off')
        axes[r, 0].set_title(label, fontsize=11, fontweight='bold', loc='left')
    plt.tight_layout()
    _save(fig, save_path, "Reconstructions")
    return fig


@torch.no_grad()
def plot_latent_traversals(decoder, z, n_components=4, n_steps=7, span=2.0,
                           save_path: Optional[str] = None):
    """
    Decode the mean latent moved along each of the top principal components of
    z (..., z_dim), from -span to +span standard deviations. Shows which visual
    factor (cart position, pole angle, ...) each principal direction encodes.
    """
    flat = z.reshape(-1, z.shape[-1])
    mean = flat.mean(axis=0)
    eig, vec = np.linalg.eigh(np.cov(flat, rowvar=False))
    order = np.argsort(eig)[::-1][:n_components]
    offsets = np.linspace(-span, span, n_steps)

    device = next(decoder.parameters()).device
    fig, axes = plt.subplots(n_components, n_steps, figsize=(1.9 * n_steps, 2.0 * n_components),
                             squeeze=False)
    for r, j in enumerate(order):
        zs = mean + offsets[:, None] * np.sqrt(eig[j]) * vec[:, j]
        imgs = decoder.decode(torch.as_tensor(zs, dtype=torch.float32, device=device))
        imgs = imgs.cpu().permute(0, 2, 3, 1).numpy()
        for c in range(n_steps):
            axes[r, c].imshow(imgs[c])
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(f'{offsets[c]:+.1f}σ', fontsize=10)
        axes[r, 0].set_ylabel(f'PC {r + 1}\n({eig[j] / eig.sum():.0%})', fontsize=10)
    fig.suptitle('Latent traversals along principal components', fontweight='bold')
    plt.tight_layout()
    _save(fig, save_path, "Latent traversals")
    return fig
