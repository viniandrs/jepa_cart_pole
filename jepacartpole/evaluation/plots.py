"""
Plots for the world-model evaluation protocol.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

from .test_set import STATE_NAMES

STATE_LABELS = ('cart position x [m]', 'cart velocity ẋ [m/s]',
                'pole angle θ [rad]', 'angular velocity θ̇ [rad/s]')
GRAY = 'gray'
ORANGE = 'orange'


def _save(fig, save_path):
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")


def _style(ax, xlabel, ylabel, title=None):
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    if title:
        ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)


def plot_state_distribution(states, dones, save_path: Optional[str] = None):
    """Histogram of each state variable over the test set."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    flat = states.reshape(-1, 4)
    for i, ax in enumerate(axes):
        ax.hist(flat[:, i], bins=60)
        _style(ax, STATE_LABELS[i], 'Frames' if i == 0 else '')
    fig.suptitle(f'Test-set state distribution ({len(flat):,} frames, {dones.sum():,} resets)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_encoder_diagnostics(diag, ceiling, save_path: Optional[str] = None):
    """Explained variance per principal component and probe ceiling per state variable."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 4))
    ev = diag['explained_variance']
    axes[0].bar(np.arange(1, len(ev) + 1), ev)
    _style(axes[0], 'Principal component', 'Explained variance ratio',
           f"Latent PCA spectrum (effective dim = {diag['effective_dim']:.2f})")
    axes[1].bar(STATE_NAMES, ceiling, color=ORANGE)
    axes[1].set_ylim(0, 1.05)
    for i, v in enumerate(ceiling):
        axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=11)
    _style(axes[1], 'State variable', 'R² (test episodes)', 'Probe ceiling on true latents')
    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_one_step(one_step, active=None, save_path: Optional[str] = None):
    """Per-dimension one-step MSE of the predictor and of the copy-last baseline."""
    dims = np.arange(len(one_step['mse_per_dim']))
    width = 0.4
    fig, ax = plt.subplots(figsize=(15, 4))
    ax.bar(dims - width / 2, one_step['mse_copy_per_dim'], width, color=GRAY, label='Copy-last baseline')
    ax.bar(dims + width / 2, one_step['mse_per_dim'], width, label='JEPA predictor')
    if active is not None:
        for d in np.nonzero(active)[0]:
            ax.axvspan(d - 0.5, d + 0.5, color=ORANGE, alpha=0.15, linewidth=0)
    ax.set_yscale('log')
    ax.set_xticks(dims)
    _style(ax, 'Latent dimension' + (' (shaded = active)' if active is not None else ''),
           'One-step MSE', 'One-step error per latent dimension')
    ax.legend()
    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_latent_rollout(roll, save_path: Optional[str] = None):
    """Skill score and NMSE of open-loop rollouts as a function of the horizon."""
    h = np.arange(1, roll['horizon'] + 1)
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
    axes[0].plot(h, roll['skill'], 'o-', linewidth=2, markersize=8, label='JEPA predictor')
    axes[0].axhline(0, linestyle='--', color=GRAY, label='Copy-last baseline (skill = 0)')
    axes[0].set_ylim(min(-0.1, np.nanmin(roll['skill']) - 0.05), 1.05)
    _style(axes[0], 'Horizon h (steps)', 'Skill = 1 − MSE / MSE_copy', 'Skill vs. horizon')
    axes[0].legend()
    axes[1].plot(h, roll['nmse'], 'o-', linewidth=2, markersize=8, label='JEPA predictor')
    axes[1].plot(h, roll['nmse_copy'], 'o--', linewidth=2, markersize=8, color=GRAY, label='Copy-last baseline')
    _style(axes[1], 'Horizon h (steps)', 'NMSE = MSE / latent variance', 'Normalized error vs. horizon')
    axes[1].legend()
    fig.suptitle(f"Open-loop latent rollouts ({roll['n']:,} segments)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_state_rollout(roll, save_path: Optional[str] = None):
    """State RMSE vs horizon for each variable: JEPA, probe ceiling, constant-state baseline."""
    h = np.arange(1, roll['horizon'] + 1)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for i, ax in enumerate(axes):
        ax.plot(h, roll['state_rmse_constant'][:, i], 'o--', linewidth=2, markersize=7, color=GRAY,
                label='Constant state')
        ax.plot(h, roll['state_rmse'][:, i], 'o-', linewidth=2, markersize=7, label='JEPA (imagined)')
        ax.plot(h, roll['state_rmse_ceiling'][:, i], 'o-', linewidth=2, markersize=7, color=ORANGE,
                label='Ceiling (true latents)')
        _style(ax, 'Horizon h', 'RMSE' if i == 0 else '', STATE_LABELS[i])
    axes[0].legend()
    fig.suptitle('Physical-state error of open-loop rollouts', fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_state_trajectories(roll, n=3, seed=0, save_path: Optional[str] = None):
    """True vs imagined state trajectories (x and θ) for a few random segments."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(roll['n'], n, replace=False)
    h = np.arange(1, roll['horizon'] + 1)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 6.5), squeeze=False)
    for col, i in enumerate(idx):
        for row, var in enumerate((0, 2)):
            ax = axes[row, col]
            ax.plot(h, roll['true_states'][i, :, var], 'o-', color='black', linewidth=2, label='True (simulator)')
            ax.plot(h, roll['ceiling_states'][i, :, var], 's-', color=ORANGE, linewidth=2, label='Probe on true latents')
            ax.plot(h, roll['pred_states'][i, :, var], 'o-', linewidth=2, label='Probe on imagined latents')
            ep, t = roll['starts'][i]
            _style(ax, 'Horizon h', STATE_LABELS[var] if col == 0 else '',
                   f'episode {ep}, t = {t}' if row == 0 else None)
    axes[0, 0].legend()
    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_counterfactual(cf, save_path: Optional[str] = None):
    """Predicted vs true action effects as a function of the horizon."""
    h = np.arange(1, cf['horizon'] + 1)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    axes[0].plot(h, cf['effect_skill'], 'o-', linewidth=2, markersize=8, label='JEPA predictor')
    axes[0].axhline(0, linestyle='--', color=GRAY, label='Action-blind model')
    axes[0].set_ylim(min(-0.1, np.nanmin(cf['effect_skill']) - 0.05), 1.05)
    _style(axes[0], 'Horizon h', 'Effect skill', 'Latent action-effect skill')
    axes[0].legend()
    axes[1].plot(h, cf['cosine'], 'o-', linewidth=2, markersize=8)
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].axhline(0, linestyle='--', color=GRAY)
    _style(axes[1], 'Horizon h', 'cos(D_pred, D_true)', 'Direction of the effect')
    for ax, var in zip(axes[2:], (0, 2)):
        ax.plot(h, cf['state_effect_true'][:, var], 'o-', color='black', linewidth=2, label='True (simulator)')
        ax.plot(h, cf['state_effect_ceiling'][:, var], 's-', color=ORANGE, linewidth=2, label='Probe on true frames')
        ax.plot(h, cf['state_effect_pred'][:, var], 'o-', linewidth=2, label='Probe on predictions')
        _style(ax, 'Horizon h', f'mean |Δ| {STATE_NAMES[var]}', f'Effect of a=1 vs a=0 on {STATE_LABELS[var]}')
    axes[2].legend()
    fig.suptitle(f"Counterfactual action test ({cf['n']:,} starts; n at h={cf['horizon']}: "
                 f"{cf['n_per_h'][-1]})", fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save(fig, save_path)
    return fig


@torch.no_grad()
def plot_decoded_rollout(decoder, roll, test_images, n_steps=8, seed=0, save_path: Optional[str] = None):
    """True frames vs decoded imagined latents for one segment (decoder-based encoders only)."""
    i = np.random.default_rng(seed).integers(roll['n'])
    ep, t = roll['starts'][i]
    n_steps = min(n_steps, roll['horizon'])
    device = next(decoder.parameters()).device
    z = torch.as_tensor(roll['z_pred'][i, :n_steps], dtype=torch.float32, device=device)
    decoded = decoder.decode(z).cpu().permute(0, 2, 3, 1).numpy()
    true = test_images[ep, t + 1:t + 1 + n_steps]
    fig, axes = plt.subplots(2, n_steps, figsize=(2.2 * n_steps, 4.6))
    for h in range(n_steps):
        axes[0, h].imshow(true[h])
        axes[1, h].imshow(decoded[h])
        axes[0, h].set_title(f't+{h + 1}', fontsize=10)
        for row in range(2):
            axes[row, h].axis('off')
    axes[0, 0].text(-8, 32, 'True', fontsize=12, fontweight='bold', rotation=90, va='center', ha='right')
    axes[1, 0].text(-8, 32, 'Imagined', fontsize=12, fontweight='bold', rotation=90, va='center', ha='right')
    fig.suptitle(f'Open-loop rollout, episode {ep}, t = {t}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save(fig, save_path)
    return fig


@torch.no_grad()
def plot_decoded_counterfactual(decoder, cf, cf_starts, cf_images, seed=0, save_path: Optional[str] = None):
    """
    True and imagined frames at the last horizon for a = 0 and a = 1
    (decoder-based encoders only). cf_starts / cf_images are the full arrays
    from the test set.
    """
    alive = np.nonzero(cf['alive'][:, -1])[0]
    i = np.random.default_rng(seed).choice(alive)
    device = next(decoder.parameters()).device
    H = cf['horizon']
    z = torch.as_tensor(cf['preds'][i, :, -1], dtype=torch.float32, device=device)
    imagined = decoder.decode(z).cpu().permute(0, 2, 3, 1).numpy()
    j = np.nonzero((cf_starts == cf['starts'][i]).all(axis=1))[0][0]
    true = cf_images[j, :, -1]
    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    for a in (0, 1):
        axes[0, a].imshow(true[a])
        axes[1, a].imshow(imagined[a])
        axes[0, a].set_title(f'True, a={a} held {H} steps', fontsize=10)
        axes[1, a].set_title(f'Imagined, a={a}', fontsize=10)
        for row in range(2):
            axes[row, a].axis('off')
    plt.tight_layout()
    _save(fig, save_path)
    return fig
