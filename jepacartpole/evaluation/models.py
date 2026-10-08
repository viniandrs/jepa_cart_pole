"""
Loading trained world models and encoding test frames.
"""

import numpy as np
import torch

from ..data_prep.latents import frames_to_tensor
from ..jepa import JEPAPredictor
from ..utils import get_checkpoint_dir, get_device, load_checkpoint
from ..vae import VAE


def load_world_model(vae_path=None, jepa_path=None, device=None):
    """
    Load a VAE encoder and a JEPA predictor from their checkpoints, building
    each model from the config stored in its checkpoint (so any latent size
    or context length is supported).

    Returns:
        tuple: (vae, predictor, vae_checkpoint, jepa_checkpoint), models in eval mode
    """
    device = device or get_device()
    vae_path = vae_path or get_checkpoint_dir() / 'vae.pth'
    jepa_path = jepa_path or get_checkpoint_dir() / 'jepa.pth'

    vae_config = torch.load(vae_path, weights_only=False, map_location='cpu')['config']
    jepa_config = torch.load(jepa_path, weights_only=False, map_location='cpu')['config']

    vae = VAE(vae_config)
    vae_ckpt = load_checkpoint(vae, vae_path, device=device)
    predictor = JEPAPredictor(jepa_config)
    jepa_ckpt = load_checkpoint(predictor, jepa_path, device=device)

    return vae.eval(), predictor.eval(), vae_ckpt, jepa_ckpt


@torch.no_grad()
def encode_frames(encoder, images, batch_size=1000, device=None):
    """
    Encode uint8 frames of any leading shape (..., H, W, C) with an encoder
    whose encode(x) returns either mu or a (mu, logvar) tuple.

    Returns:
        tuple: (mu, logvar) numpy arrays of shape (..., z_dim); logvar is None
               if the encoder is deterministic
    """
    device = device or next(encoder.parameters()).device
    encoder.eval()
    lead_shape = images.shape[:-3]
    flat = images.reshape(-1, *images.shape[-3:])

    mus, logvars = [], []
    for start in range(0, len(flat), batch_size):
        out = encoder.encode(frames_to_tensor(flat[start:start + batch_size], device))
        mu, logvar = out if isinstance(out, tuple) else (out, None)
        mus.append(mu.cpu().numpy())
        if logvar is not None:
            logvars.append(logvar.cpu().numpy())

    mu = np.concatenate(mus).reshape(*lead_shape, -1)
    logvar = np.concatenate(logvars).reshape(*lead_shape, -1) if logvars else None
    return mu, logvar
