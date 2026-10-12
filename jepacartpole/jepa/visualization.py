"""
Visualization utilities for JEPA predictor training.

Evaluation plots (rollouts, decoded predictions, counterfactuals) live in
jepacartpole.evaluation.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt


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
