"""
Full JEPA world model: a pixel encoder and a latent predictor trained jointly,
with an EMA target encoder and VICReg variance/covariance regularisation to
prevent representation collapse.
"""

import copy

import torch
import torch.nn as nn

from ..config import ConfigFullJEPA, ConfigJEPA
from ..jepa import JEPAPredictor
from ..utils import get_checkpoint_dir, get_device, load_checkpoint
from ..vae.encoder import make_conv_encoder


class JEPAEncoder(nn.Module):
    """
    Deterministic encoder with the World Models conv stack (same as the VAE)
    followed by a linear head: (C, 64, 64) -> z_dim.
    """
    def __init__(self, config: ConfigFullJEPA):
        super(JEPAEncoder, self).__init__()
        self.latent_dim = config.latent_dim
        self.conv = make_conv_encoder(config.image_channels)  # output: (1024)
        self.head = nn.Linear(256 * 2 * 2, config.latent_dim)

    def forward(self, x):
        return self.head(self.conv(x))

    def encode(self, x):
        return self(x)


class FullJEPA(nn.Module):
    """
    Context encoder, EMA target encoder and MLP predictor.

    The predictor is the same JEPAPredictor used on top of the VAE, so the
    two world models differ only in how the encoder is trained.
    encode(x) uses the target encoder: it is the space the predictor is
    trained to output and the one used for evaluation.
    """
    def __init__(self, config: ConfigFullJEPA):
        super(FullJEPA, self).__init__()
        self.config = config
        self.encoder = JEPAEncoder(config)
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.predictor = JEPAPredictor(ConfigJEPA(
            z_dim=config.latent_dim,
            action_dim=config.action_dim,
            context_len=config.context_len,
            hidden_dim=config.hidden_dim,
        ))
        # VICReg expander: variance/covariance are applied on g(z) so that z
        # itself can stay low-dimensional (Bardes et al., 2022)
        e = getattr(config, 'expander_dim', 0)
        self.expander = nn.Sequential(
            nn.Linear(config.latent_dim, e), nn.BatchNorm1d(e), nn.ReLU(), nn.Linear(e, e)
        ) if e > 0 else nn.Identity()

    @property
    def context_len(self):
        return self.predictor.context_len

    @torch.no_grad()
    def update_target(self, momentum):
        """EMA update of the target encoder: theta_bar <- m * theta_bar + (1 - m) * theta."""
        for p_t, p in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            p_t.mul_(momentum).add_(p.detach(), alpha=1 - momentum)

    @torch.no_grad()
    def encode(self, x):
        return self.target_encoder(x)


def vicreg_terms(z, gamma=1.0, eps=1e-4):
    """
    VICReg variance and covariance terms (Bardes et al., 2022) on a batch of
    embeddings z (n, d).

    variance:   mean_j relu(gamma - sqrt(Var(z_j) + eps))
    covariance: sum of squared off-diagonal entries of Cov(z), divided by d

    Returns:
        tuple: (var_loss, cov_loss)
    """
    n, d = z.shape
    z = z - z.mean(dim=0)
    std = torch.sqrt(z.var(dim=0) + eps)
    var_loss = torch.relu(gamma - std).mean()
    cov = (z.T @ z) / (n - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = off_diag.pow(2).sum() / d
    return var_loss, cov_loss


def ema_momentum(step, total_steps, start, end):
    """Cosine schedule of the target-encoder momentum from start to end (I-JEPA)."""
    progress = min(step / max(total_steps, 1), 1.0)
    return end - (end - start) * (1 + torch.cos(torch.tensor(torch.pi * progress)).item()) / 2


def load_full_jepa(path=None, device=None):
    """
    Load a FullJEPA from its checkpoint, built from the stored config.

    Returns:
        tuple: (model in eval mode, checkpoint dict)
    """
    device = device or get_device()
    path = path or get_checkpoint_dir() / 'full_jepa.pth'
    config = torch.load(path, weights_only=False, map_location='cpu')['config']
    model = FullJEPA(config)
    checkpoint = load_checkpoint(model, path, device=device)
    return model.eval(), checkpoint
