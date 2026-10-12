# Findings Log — VAE + JEPA World Model

Running record of conclusions to revisit later. Newest section first. Numbers come from the executed notebooks named in each section.

---

## 2026-10-09 — Evaluation protocol and v1 baseline (`vae-z32`)

**State of the code:** v1 VAE (`latent_dim=32`, `kl_tolerance=0.5`) retrained on GPU; JEPA predictor `context_len=2`, trained on sampled latents (`sample_latents=True`), 20 epochs. Notebook 3 = training only; notebook 4 = evaluation protocol (`jepacartpole/evaluation/`). Scorecard: `results/eval/vae-z32.json` (gitignored — numbers copied below).

### Evaluation methodology (decisions and why)

- **Held-out test set** `data/eval/test_set.h5`: 100 episodes × 500 steps, same policy as training, seeds `100000+i` (training: `42+i`). Stores true states and simulator counterfactuals. Episodes 0–49 fit the probes; 50–99 are evaluated.
- **Never compare raw latent MSE across encoders.** Latent metrics are relative: skill = 1 − MSE/MSE_copy-last, NMSE = MSE / latent variance.
- **Primary cross-encoder metric: physical-state RMSE vs horizon**, via linear probes `[z_t, z_{t-1}] → (x, ẋ, θ, θ̇)`, always shown with a *ceiling* (probe on true latents) and a *constant-state* baseline.
- **Probe must use the principal subspace (99% of latent variance, `PairPCA`).** With all 32 dims the probe exploits near-constant directions; the predictor's errors there are as large as the signal, and the JEPA looked *worse* than the constant-state baseline for velocities (one-step ẋ RMSE 0.575 vs 0.327). With 3 PCs: 0.268 (ceiling 0.269). The ceiling is a reference level, not a strict bound (a conditional-mean predictor can average out per-frame encoding noise).
- **CartPole uses explicit Euler integration**: positions at t+1 use the *old* velocity, so frame t+1 does **not** depend on a_t (counterfactual frames at h=1 are pixel-identical). Action effects appear from t+2 and grow ~quadratically (Δθ between actions: 0.012 rad at h=2, 0.07 at h=4, 0.33 at h=8).
- **Data policy holds actions 20 steps → a_t ≈ a_{t-1} ~95% of the time.** Comparing predictions with the observed future rewards this correlation. Hence the action test uses true counterfactuals (both actions held for 8 steps from the same simulator state; branches masked after the pole falls, ≥179 starts at h=8 in the eval half).
- **Rollout selection effect**: segments with 8 reset-free steps mostly start 1–2 steps after a reset (slow motion), so h=1 of the rollout test is harder vs copy-last than the all-transitions one-step test. Comparable across encoders because segments are identical.

### v1 results

| Test | Result |
|---|---|
| Encoder | 4/32 active dims, effective dim 2.09, probe uses 3 PCs (99.5%). Ceiling R²: x 0.991, ẋ 0.934, θ 0.993, θ̇ 0.941 |
| One-step (all transitions) | latent skill 0.887, NMSE 0.0085 (copy-last 0.075). State RMSE: x 7.9 mm (ceiling 6.9, constant 19.2); θ 0.010 rad (0.008 / 0.028); ẋ 0.268 (0.269 / 0.365); θ̇ 0.376 (0.380 / 0.550) |
| Action (counterfactual) | Effect direction right from h=2 (cosine > 0.9) but applied **one step early** (predicts Δθ 0.0095 rad at h=1 where the truth is 0). Effect skill −3.10 (h=2), −0.19 (h=3), 0.44 (h=4), 0.79 (h=8) |
| Latent rollouts | skill 0.42 (h=1) → 0.71 (h=4) → 0.83 (h=8); NMSE 0.45 at h=8 vs 2.62 copy-last |
| Physical rollouts, h=8 | x 4.7 cm (ceiling 1.5, constant 12.3); θ 0.069 rad (0.016 / 0.170); ≈ 66–70% of the gap between constant and ceiling closed. Error grows ~8× from h=1 to h=8 vs ~3× for the ceiling |

### Conclusions

1. **The encoder is not the bottleneck** for v1: 3 principal components hold the state linearly; one-step predictions are nearly as informative as true latents.
2. **Main weakness: action timing.** The predictor attributes the effect of a_{t−1} to a_t (learnable only because of the held-action policy). Invisible to accuracy metrics; exposed only by counterfactuals.
3. **Compounding error** dominates beyond a few steps; the 8-step horizon is limited by the data (pole falls every ~10 steps under random actions).

### Open ideas (not tried yet)

- Give the predictor the previous action(s) (`a_{t-1}, a_t`) or a recurrent state; or collect data with actions that change more often (shorter `action_interval`) to break the a_t ≈ a_{t−1} correlation.
- Train the predictor on μ (`sample_latents=False`): ~2× lower val error than sampled z for v1 (0.00032 vs 0.00061, 20 epochs, earlier run), because 28/32 dims are pure noise when sampled.
- Data with longer sub-episodes (e.g. a stabilising policy mixed with random actions) to allow horizons > 8.
- Evaluate the ResNet encoder with the same notebook 4 (new `RUN_NAME`) and compare scorecards.

---

## 2026-10-08 — Latent dimensionality experiments (VAE v1 vs v2)

These runs were archived and later removed from `registry/` at the user's request; the conclusions are kept here. Numbers use the *older* evaluation (JEPA validation split, ratio to copy-last), not the protocol above.

- **v1 (`latent_dim=32`, KL floor 16 nats):** only 4/32 dims active (σ ≈ 0.01 on the main ones); the other 28 sit at the prior (σ ≈ 0.95). Consequences for sampled-latent JEPA training: train loss ≈ noise floor (0.809 vs 0.808), ~30% of val error in inactive dims (where JEPA is worse than copy-last). JEPA 8.6× below copy-last (sampled z), 16.6× (trained on μ). VAE PSNR 52.3 dB.
- **v2 (`latent_dim=4`, `kl_tolerance=0.5` → floor 2 nats):** VAE kept exactly **2** active dims (z₃ ≈ cart x, corr 1.00; θ spread over z₀/z₃), probes as good as v1, PSNR 46.6 dB. But the posterior σ grew to 0.34 / 0.10, i.e. 2.1× / 0.41× the RMS one-step latent change, burying velocity information when sampling. JEPA only 4.0× below copy-last (sampled z), 9.9× (μ); rollouts ≈ copy-last up to h=4.
- **Lesson:** shrinking `latent_dim` while keeping `kl_tolerance=0.5` changes two things at once (dimension *and* information budget 16 → 2 nats). CartPole needs only ~2 dims, but they must be **precise** (σ small relative to per-step motion). Candidate to test: `latent_dim=2` with `kl_tolerance=8` (keeps the 16-nat budget), and/or μ-trained predictor. Re-evaluate any such change with notebook 4.

## Practical notes

- VAE training: ~11 min on a T4, ~13 min/epoch (~2.2 h) on the 4-CPU Studio. Run long notebooks detached (`setsid nohup jupyter nbconvert --execute --inplace …`) — Studio restarts wipe `/tmp` and kill attached jobs.
- Test-set generation (CPU): ~25 min for 100 episodes + 6144 counterfactual starts; file 1.7 GB.
