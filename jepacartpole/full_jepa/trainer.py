"""
Full JEPA training (encoder + predictor jointly) and training of the
qualitative decoder on frozen JEPA latents.
"""

from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from ..jepa.dataset import split_episodes
from ..utils import save_checkpoint, get_checkpoint_dir, get_data_dir, set_seed
from .dataset import FrameWindowDataset, load_episodes, frames_to_device
from .model import vicreg_terms, ema_momentum


@torch.no_grad()
def encode_episodes(encoder, images, device, batch_size=1000):
    """Encode uint8 frames (N, T, H, W, C) -> numpy latents (N, T, z_dim)."""
    encoder.eval()
    flat = images.reshape(-1, *images.shape[2:])
    z = torch.cat([encoder(frames_to_device(flat[s:s + batch_size], device)).cpu()
                   for s in range(0, len(flat), batch_size)])
    return z.reshape(*images.shape[:2], -1).numpy()


def effective_dim(z):
    """PCA participation ratio (sum lambda)^2 / sum lambda^2 of latents (..., z_dim)."""
    eig = np.clip(np.linalg.eigvalsh(np.cov(z.reshape(-1, z.shape[-1]), rowvar=False)), 0, None)
    return float(eig.sum() ** 2 / (eig ** 2).sum())


@torch.no_grad()
def evaluate_full_jepa(model, val_data, device):
    """
    Validation metrics in the evaluation setting (every latent from the
    target encoder, as model.encode does).

    Returns:
        dict with
          - skill: 1 - MSE(pred) / MSE(copy-last) on one-step transitions
          - skill_online: same with the context from the online encoder, as
            during training (the gap to `skill` comes from encoder_gap)
          - pred_mse, copy_mse: raw latent MSEs (only comparable within a run)
          - std: mean per-dimension std of the latents
          - effective_dim: PCA participation ratio of the latents
          - encoder_gap: ||Enc(x) - TargetEnc(x)||^2 / total latent variance
    """
    z = encode_episodes(model.target_encoder, val_data.images, device)
    z_online = encode_episodes(model.encoder, val_data.images, device)

    k = model.context_len
    ep, t = val_data.index[:, :1], val_data.index[:, 1:]
    z_ctx = torch.from_numpy(z[ep, t + np.arange(-k + 1, 1)]).to(device)
    z_next = torch.from_numpy(z[ep[:, 0], t[:, 0] + 1]).to(device)
    actions = torch.from_numpy(val_data.actions[ep[:, 0], t[:, 0], None]).to(device)

    z_ctx_online = torch.from_numpy(z_online[ep, t + np.arange(-k + 1, 1)]).to(device)

    def predict(ctx):
        return torch.cat([model.predictor(ctx[s:s + 8192], actions[s:s + 8192])
                          for s in range(0, len(ctx), 8192)])

    model.predictor.eval()
    pred_mse = F.mse_loss(predict(z_ctx), z_next).item()
    pred_mse_online = F.mse_loss(predict(z_ctx_online), z_next).item()
    model.predictor.train()
    copy_mse = F.mse_loss(z_ctx[:, -1], z_next).item()

    flat = z.reshape(-1, z.shape[-1])
    total_var = flat.var(axis=0).sum()
    return {
        'skill': 1 - pred_mse / copy_mse,
        'skill_online': 1 - pred_mse_online / copy_mse,
        'pred_mse': pred_mse,
        'copy_mse': copy_mse,
        'std': float(flat.std(axis=0).mean()),
        'effective_dim': effective_dim(z),
        'encoder_gap': float(((z_online - z) ** 2).reshape(-1, z.shape[-1]).sum(axis=1).mean() / total_var),
    }


def train_full_jepa(
    model: nn.Module,
    config,
    h5_path: Optional[str] = None,
    epochs: Optional[int] = None,
    checkpoint_dir: Optional[str] = None,
    device: Optional[torch.device] = None,
    seed: int = 42,
    episodes: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> dict:
    """
    Train the encoder and the predictor of a FullJEPA jointly.

    For every window (x_{t-k+1..t}, a_t, x_{t+1}):
        z_ctx    = Enc(x_{t-k+1..t})                (gradients flow into Enc)
        z_hat    = Pred(z_ctx, a_t)
        z_target = TargetEnc(x_{t+1})               (no gradient, EMA of Enc)
        loss     = pred_weight * MSE(z_hat, z_target)
                 + var_weight * Var(z_ctx) + cov_weight * Cov(z_ctx)   (VICReg)
    After every optimiser step the target encoder is updated as an EMA of
    the encoder, with a momentum following a cosine schedule from
    config.ema_momentum to config.ema_momentum_end.

    Episodes are split into train/validation sets (config.val_fraction).
    Saves full_jepa_latest.pth after every epoch and full_jepa.pth at the end.

    Args:
        model: FullJEPA instance
        config: ConfigFullJEPA
        h5_path: Frame dataset (default: data/vae/vae_data.h5)
        epochs: Number of epochs (default: config.epochs)
        checkpoint_dir: Directory for checkpoints (default: project checkpoints/)
        device: torch.device (defaults to the model's device)
        seed: Random seed (also fixes the episode split)
        episodes: Optional subset of episodes to use (for quick tests)
        verbose: Print progress information

    Returns:
        dict of per-epoch histories (train loss terms and validation metrics)
        plus train_episodes / val_episodes
    """
    set_seed(seed)
    rng = np.random.default_rng(seed)
    epochs = epochs or config.epochs
    device = device or next(model.parameters()).device
    h5_path = h5_path or get_data_dir('vae') / 'vae_data.h5'
    checkpoint_dir = get_checkpoint_dir() if checkpoint_dir is None else Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if episodes is None:
        with h5py.File(h5_path, 'r') as h5f:
            episodes = np.arange(h5f['images'].shape[0])
    train_idx, val_idx = split_episodes(len(episodes), config.val_fraction, seed)
    train_episodes, val_episodes = np.asarray(episodes)[train_idx], np.asarray(episodes)[val_idx]

    train_data = FrameWindowDataset(h5_path, config.context_len, train_episodes)
    val_data = FrameWindowDataset(h5_path, config.context_len, val_episodes)
    steps_per_epoch = train_data.num_batches(config.batch_size)
    total_steps = epochs * steps_per_epoch

    if verbose:
        print(f"\n{'='*80}")
        print(f"FULL JEPA TRAINING (encoder + predictor)")
        print(f"{'='*80}")
        print(f"Dataset: {h5_path}")
        print(f"Train: {len(train_episodes)} episodes, {len(train_data):,} windows")
        print(f"Val:   {len(val_episodes)} episodes, {len(val_data):,} windows")
        print(f"Config: {config}")
        print(f"Device: {device}")
        print(f"{'='*80}\n")

    model.to(device)
    params = [p for name, p in model.named_parameters() if not name.startswith('target_encoder.')]
    optimizer = AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)

    keys = ('loss', 'pred_loss', 'var_loss', 'cov_loss')
    history = {f'train_{key}': [] for key in keys}
    history.update({f'val_{key}': [] for key in ('skill', 'skill_online', 'pred_mse', 'copy_mse', 'std', 'effective_dim', 'encoder_gap')})
    history['momentum'] = []

    val = evaluate_full_jepa(model, val_data, device)
    history['initial_val'] = val
    if verbose:
        print(f"Before training: val skill={val['skill']:.4f}  std={val['std']:.4f}  "
              f"eff. dim={val['effective_dim']:.2f}")

    k = config.context_len
    step = 0
    for epoch in range(epochs):
        model.encoder.train()
        model.predictor.train()
        model.expander.train()
        model.target_encoder.eval()
        sums = dict.fromkeys(keys, 0.0)
        batches = train_data.batches(config.batch_size, device, shuffle=True, rng=rng)
        progress = tqdm(batches, total=steps_per_epoch, desc=f"Epoch {epoch + 1}/{epochs}",
                        leave=False) if verbose else batches

        for frames, actions in progress:
            B = frames.shape[0]
            z_ctx = model.encoder(frames[:, :k].flatten(0, 1)).view(B, k, -1)
            with torch.no_grad():
                z_target = model.target_encoder(frames[:, k])
            z_pred = model.predictor(z_ctx, actions)

            pred_loss = F.mse_loss(z_pred, z_target)
            var_loss, cov_loss = vicreg_terms(model.expander(z_ctx.flatten(0, 1)), config.var_gamma)
            loss = (config.pred_weight * pred_loss + config.var_weight * var_loss
                    + config.cov_weight * cov_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            m = ema_momentum(step, total_steps, config.ema_momentum, config.ema_momentum_end)
            model.update_target(m)
            step += 1

            for key, value in zip(keys, (loss, pred_loss, var_loss, cov_loss)):
                sums[key] += value.item()
            if verbose:
                progress.set_postfix({'loss': f"{loss.item():.4f}", 'pred': f"{pred_loss.item():.4f}"})

        for key in keys:
            history[f'train_{key}'].append(sums[key] / steps_per_epoch)
        history['momentum'].append(m)
        val = evaluate_full_jepa(model, val_data, device)
        for key, value in val.items():
            history[f'val_{key}'].append(value)

        save_checkpoint(model, checkpoint_dir / 'full_jepa_latest.pth', epoch=epoch,
                        optimizer_state_dict=optimizer.state_dict(), config=config,
                        history=history, val_episodes=val_episodes)

        if verbose:
            print(f"Epoch {epoch + 1}/{epochs}: loss={history['train_loss'][-1]:.4f} "
                  f"(pred={history['train_pred_loss'][-1]:.5f} var={history['train_var_loss'][-1]:.4f} "
                  f"cov={history['train_cov_loss'][-1]:.4f})  val skill={val['skill']:.4f}  "
                  f"std={val['std']:.3f}  eff. dim={val['effective_dim']:.2f}  m={m:.4f}")

    save_checkpoint(model, checkpoint_dir / 'full_jepa.pth', epoch=epochs - 1,
                    optimizer_state_dict=optimizer.state_dict(), config=config,
                    history=history, val_episodes=val_episodes)

    if verbose:
        print(f"\n{'='*80}")
        print(f"TRAINING COMPLETE — model saved to {checkpoint_dir / 'full_jepa.pth'}")
        print(f"Final val skill vs copy-last: {history['val_skill'][-1]:.4f}")
        print(f"{'='*80}\n")

    history.update(train_episodes=train_episodes, val_episodes=val_episodes)
    return history


def train_latent_decoder(
    decoder: nn.Module,
    encoder: nn.Module,
    config,
    train_episodes,
    val_episodes,
    h5_path: Optional[str] = None,
    epochs: Optional[int] = None,
    checkpoint_dir: Optional[str] = None,
    device: Optional[torch.device] = None,
    seed: int = 42,
    verbose: bool = True,
    **checkpoint_metadata,
) -> dict:
    """
    Train a LatentDecoder to reconstruct frames from frozen latents
    z = encoder.encode(x). The encoder is never updated (latents are
    computed under no_grad). Loss: per-sample sum of squared pixel errors,
    as the VAE reconstruction term.

    Saves latent_decoder_latest.pth after every epoch and latent_decoder.pth
    at the end (extra keyword arguments are stored in the checkpoint).

    Returns:
        dict with epoch_train_losses, epoch_val_losses (per-sample SSE) and
        epoch_val_psnr (dB)
    """
    set_seed(seed)
    rng = np.random.default_rng(seed)
    epochs = epochs or config.epochs
    device = device or next(decoder.parameters()).device
    h5_path = h5_path or get_data_dir('vae') / 'vae_data.h5'
    checkpoint_dir = get_checkpoint_dir() if checkpoint_dir is None else Path(checkpoint_dir)

    train_images = load_episodes(h5_path, train_episodes, keys=('images',))['images']
    val_images = load_episodes(h5_path, val_episodes, keys=('images',))['images']
    train_flat = train_images.reshape(-1, *train_images.shape[2:])
    val_flat = val_images.reshape(-1, *val_images.shape[2:])

    if verbose:
        print(f"Decoder training: {len(train_flat):,} train frames, {len(val_flat):,} val frames, "
              f"{epochs} epochs, device {device}")

    encoder.eval()
    decoder.to(device)
    optimizer = AdamW(decoder.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    @torch.no_grad()
    def validate():
        decoder.eval()
        sse, mse = 0.0, 0.0
        for s in range(0, len(val_flat), 1000):
            x = frames_to_device(val_flat[s:s + 1000], device)
            err = (decoder.decode(encoder.encode(x)) - x).pow(2)
            sse += err.sum().item()
            mse += err.mean(dim=(1, 2, 3)).sum().item()
        decoder.train()
        return sse / len(val_flat), mse / len(val_flat)

    history = {'epoch_train_losses': [], 'epoch_val_losses': [], 'epoch_val_psnr': []}
    num_batches = (len(train_flat) + config.batch_size - 1) // config.batch_size
    for epoch in range(epochs):
        decoder.train()
        total = 0.0
        order = rng.permutation(len(train_flat))
        batches = range(0, len(order), config.batch_size)
        progress = tqdm(batches, total=num_batches, desc=f"Epoch {epoch + 1}/{epochs}",
                        leave=False) if verbose else batches
        for s in progress:
            x = frames_to_device(train_flat[np.sort(order[s:s + config.batch_size])], device)
            with torch.no_grad():
                z = encoder.encode(x)
            loss = (decoder.decode(z) - x).pow(2).sum(dim=(1, 2, 3)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()

        val_sse, val_mse = validate()
        history['epoch_train_losses'].append(total / num_batches)
        history['epoch_val_losses'].append(val_sse)
        history['epoch_val_psnr'].append(float(10 * np.log10(1.0 / val_mse)))

        save_checkpoint(decoder, checkpoint_dir / 'latent_decoder_latest.pth', epoch=epoch,
                        latent_dim=decoder.latent_dim, image_channels=decoder.image_channels,
                        history=history, **checkpoint_metadata)
        if verbose:
            print(f"Epoch {epoch + 1}/{epochs}: train={history['epoch_train_losses'][-1]:.2f}  "
                  f"val={val_sse:.2f}  val PSNR={history['epoch_val_psnr'][-1]:.2f} dB")

    save_checkpoint(decoder, checkpoint_dir / 'latent_decoder.pth', epoch=epochs - 1,
                    latent_dim=decoder.latent_dim, image_channels=decoder.image_channels,
                    history=history, **checkpoint_metadata)
    if verbose:
        print(f"Decoder saved to: {checkpoint_dir / 'latent_decoder.pth'}")
    return history
