# v1 — VAE with `latent_dim = 32`

Archived on 2026-10-08. Code at commit `46ec770` plus the uncommitted JEPA refactor of the same day (`jepacartpole/jepa/`, `jepacartpole/data_prep/latents.py`).

## Notebooks

| Notebook | Content |
|---|---|
| [2-vae/1-vae_training.ipynb](2-vae/1-vae_training.ipynb) | VAE training (run on GPU, 10 min) |
| [2-vae/2-vae_evaluation.ipynb](2-vae/2-vae_evaluation.ipynb) | Reconstruction, posterior collapse, traversals, state probes |
| [1-data_prep/2-latent_encoding.ipynb](1-data_prep/2-latent_encoding.ipynb) | Encoding of the 250k frames into `(μ, log σ²)` |
| [2-vae/3-vae_jepa.ipynb](2-vae/3-vae_jepa.ipynb) | JEPA predictor training, rollouts and latent-dimensionality analysis |

## Configuration

- `ConfigVAE(latent_dim=32, beta=1.0, kl_tolerance=0.5, batch_size=100, epochs=10, learning_rate=1e-4, weight_decay=1e-6)` → KL floor 16 nats
- `ConfigJEPA(z_dim=32, action_dim=1, context_len=2, hidden_dim=128, batch_size=128, epochs=20, learning_rate=1e-3, sample_latents=True, weight_decay=0.0, val_fraction=0.1)`
- Data: 500 episodes × 500 steps, random action held for 20 steps, 64×64 frames.

## Key Results

**VAE**

| Metric | Value |
|---|---|
| Final recon loss (SSE / image) | 0.0846 |
| Final KL | 15.4 nats (floor 16) |
| Reconstruction PSNR | 52.3 dB |
| Active dims (KL > 0.05) | 4 / 32 (1, 5, 27, 29) |

Linear probe R² (test episodes):

| State | from z_t | from [z_t, z_{t-1}] |
|---|---|---|
| cart position x | 0.996 | 0.998 |
| cart velocity ẋ | 0.722 | 0.956 |
| pole angle θ | 0.996 | 0.997 |
| angular velocity θ̇ | 0.760 | 0.959 |

**JEPA predictor** (validation = 50 held-out episodes, MSE on μ)

| Metric | Value |
|---|---|
| Copy-last baseline | 0.0052 |
| Best val MSE, trained on sampled z | 0.00061 (8.6× below baseline) |
| Best val MSE, trained on μ (ablation) | 0.00032 |
| Train loss (sampled z) / noise floor | 0.809 / 0.808 |
| Share of val error in inactive dims | ~30% |
| Open-loop rollout, 8 steps | ~4× below baseline |

## Why It Was Superseded

The VAE leaves 28 of its 32 dimensions at the prior (σ ≈ 0.95). For the JEPA this means 28 useless inputs and 28 noisy targets: the sampled-latent training loss is 99.9% sampling noise, training on μ is 2× better, and ~30% of the predictor's validation error comes from dimensions that carry no information. Next version: `latent_dim = 4` (the number of active dimensions), keeping `kl_tolerance = 0.5`.
