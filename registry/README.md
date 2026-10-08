# Model Registry

Executed notebooks of superseded model versions, kept so that results can be compared across versions and cited in the thesis.

## Convention

- Whenever a model is improved, the **executed** notebooks of the current version are copied here **before** they are re-run for the new version. `notebooks/` always holds the current model.
- One folder per superseded version: `vN-<short-name>/` (e.g. `v1-vae-z32`), mirroring the `notebooks/` layout. Only the notebooks of stages that depend on the model are copied (data collection is model-independent).
- Each folder has a `README.md` with the date, the configuration, the key results and the reason the version was superseded.
- Checkpoints and data are **not** archived (they are large and gitignored). The notebooks are records of the outputs: they are not guaranteed to re-run against later versions of the `jepacartpole` package.

## Versions

| Version | Encoder | Superseded because | Date |
|---|---|---|---|
| [v1-vae-z32](v1-vae-z32/README.md) | World Models ConvVAE, `latent_dim=32` | Only 4/32 latent dims active; unused dims dominate sampled-latent JEPA training | 2026-10-08 |

**Current version (in `notebooks/`):** v2 — ConvVAE with `latent_dim=4`, `kl_tolerance=0.5` (2-nat KL floor). Only 2 dims active; latents less precise than v1, JEPA 4.0× below baseline (sampled z) vs 8.6× for v1. See the analysis at the end of `notebooks/2-vae/3-vae_jepa.ipynb`.
