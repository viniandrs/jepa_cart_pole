"""
Encoder-agnostic evaluation metrics for latent world models.

All functions work on numpy latents of the test set, z with shape
(N, T, z_dim), and a predictor f(z_ctx, action) -> z_next (JEPAPredictor
interface). Latent-space metrics are normalized by baselines or by the latent
variance; physical-state metrics go through linear probes (see probes.py).
"""

import numpy as np
import torch

from .probes import predict_states, r2_per_variable


# ----------------------------------------------------------------------------
# Segments and rollouts
# ----------------------------------------------------------------------------

def valid_starts(dones, context_len, horizon, episodes=None):
    """
    Start points (episode, t) such that the context frames t-k+1..t and the
    targets t+1..t+horizon all belong to the same sub-episode, i.e.
    dones[t-k+1 : t+horizon] contains no reset.

    Returns:
        (n, 2) int array of (episode, t)
    """
    num_eps, T = dones.shape
    k = context_len
    csum = np.concatenate([np.zeros((num_eps, 1), int), np.cumsum(dones, axis=1)], axis=1)
    t = np.arange(k - 1, T - horizon)
    resets = csum[:, t + horizon] - csum[:, t - k + 1]
    ep_idx, t_idx = np.nonzero(resets == 0)
    starts = np.stack([ep_idx, t[t_idx]], axis=1)
    if episodes is not None:
        starts = starts[np.isin(starts[:, 0], episodes)]
    return starts


def gather_segments(z, actions, starts, context_len, horizon):
    """
    Returns:
        z_ctx (n, k, z_dim), acts (n, horizon), z_future (n, horizon, z_dim)
    """
    ep, t = starts[:, :1], starts[:, 1:]
    z_ctx = z[ep, t + np.arange(-context_len + 1, 1)]
    z_future = z[ep, t + np.arange(1, horizon + 1)]
    acts = actions[ep, t + np.arange(horizon)]
    return z_ctx, acts, z_future


@torch.no_grad()
def rollout(predictor, z_ctx, actions, device, batch_size=8192):
    """
    Open-loop rollout: predictions are fed back as context, actions are given.

    Args:
        z_ctx: (n, k, z_dim) initial context (numpy or tensor)
        actions: (n, horizon) actions taken from the last context step on

    Returns:
        (n, horizon, z_dim) numpy array of imagined latents
    """
    predictor.eval()
    out = []
    for s in range(0, len(z_ctx), batch_size):
        ctx = torch.as_tensor(z_ctx[s:s + batch_size], dtype=torch.float32, device=device)
        acts = torch.as_tensor(actions[s:s + batch_size], dtype=torch.float32, device=device)
        preds = []
        for h in range(acts.shape[1]):
            z_next = predictor(ctx, acts[:, h:h + 1])
            preds.append(z_next)
            ctx = torch.cat([ctx[:, 1:], z_next.unsqueeze(1)], dim=1)
        out.append(torch.stack(preds, dim=1).cpu().numpy())
    return np.concatenate(out)


def _latent_variance(z):
    """Mean per-dimension variance of the latents (total variance / z_dim)."""
    return z.reshape(-1, z.shape[-1]).var(axis=0).mean()


# ----------------------------------------------------------------------------
# Test 0: encoder diagnostics
# ----------------------------------------------------------------------------

def encoder_diagnostics(z, logvar=None, kl_threshold=0.05):
    """
    Properties of the latent space itself (no predictor involved).

    Returns:
        dict with latent_dim, effective_dim (PCA participation ratio) and, for
        stochastic encoders, active_dims (mean KL > kl_threshold), mean_sigma
        and noise_floor (mean posterior variance)
    """
    flat = z.reshape(-1, z.shape[-1])
    eig = np.clip(np.linalg.eigvalsh(np.cov(flat, rowvar=False)), 0, None)
    out = {
        'latent_dim': int(z.shape[-1]),
        'effective_dim': float(eig.sum() ** 2 / (eig ** 2).sum()),
        'explained_variance': np.sort(eig)[::-1] / eig.sum(),
    }
    if logvar is not None:
        lv = logvar.reshape(-1, logvar.shape[-1])
        var = np.exp(lv)
        kl = (0.5 * (flat ** 2 + var - 1 - lv)).mean(axis=0)
        out.update(
            kl_per_dim=kl,
            active_dims=int((kl > kl_threshold).sum()),
            mean_sigma_per_dim=np.sqrt(var).mean(axis=0),
            noise_floor=float(var.mean()),
        )
    return out


# ----------------------------------------------------------------------------
# Test 1: one-step prediction
# ----------------------------------------------------------------------------

def one_step_metrics(predictor, z, actions, dones, context_len, device, episodes=None):
    """
    One-step prediction error on every valid transition of the test set.

    skill = 1 - MSE / MSE_copy, where the copy-last baseline predicts z_{t+1} = z_t
    nmse  = MSE / mean latent variance (sum over dims of the error / total variance)

    Returns:
        dict of scalars and per-dimension arrays
    """
    starts = valid_starts(dones, context_len, 1, episodes)
    z_ctx, acts, z_fut = gather_segments(z, actions, starts, context_len, 1)
    pred = rollout(predictor, z_ctx, acts, device)[:, 0]
    true, last = z_fut[:, 0], z_ctx[:, -1]

    se = ((pred - true) ** 2).mean(axis=0)
    se_copy = ((last - true) ** 2).mean(axis=0)
    var = z.reshape(-1, z.shape[-1]).var(axis=0)
    return {
        'n': len(starts),
        'mse': float(se.mean()),
        'mse_copy': float(se_copy.mean()),
        'skill': float(1 - se.sum() / se_copy.sum()),
        'nmse': float(se.sum() / var.sum()),
        'nmse_copy': float(se_copy.sum() / var.sum()),
        'mse_per_dim': se,
        'mse_copy_per_dim': se_copy,
        'r2_per_dim': 1 - se / var,
    }


# ----------------------------------------------------------------------------
# Tests 3 and 4: open-loop rollouts in latent and physical-state space
# ----------------------------------------------------------------------------

def rollout_metrics(predictor, z, actions, dones, states, probe, context_len,
                    horizon, device, episodes=None):
    """
    Open-loop rollouts of `horizon` steps from every valid start.

    Latent space (per horizon h): skill vs copy-last and NMSE.
    State space (per horizon h, per variable): RMSE and R2 of the probe
    applied to the imagined latents [z_hat_{t+h}, z_hat_{t+h-1}], compared to
      - ceiling:  the probe applied to the true future latents
      - constant: the probe state at time t held constant (copy-last in state space)

    Returns:
        dict of per-horizon arrays plus the raw arrays (for plots)
    """
    k_sel = max(context_len, 2)   # z_{t-1} is needed for the state at time t
    starts = valid_starts(dones, k_sel, horizon, episodes)
    z_ctx, acts, z_fut = gather_segments(z, actions, starts, k_sel, horizon)
    pred = rollout(predictor, z_ctx[:, -context_len:], acts, device)
    last = z_ctx[:, -1:]

    se = ((pred - z_fut) ** 2).sum(axis=2).mean(axis=0)            # (H,)
    se_copy = ((last - z_fut) ** 2).sum(axis=2).mean(axis=0)
    total_var = _latent_variance(z) * z.shape[-1]

    # Physical state through the probe
    ep, t = starts[:, :1], starts[:, 1:]
    true_states = states[ep, t + np.arange(1, horizon + 1)]          # (n, H, 4)
    pred_full = np.concatenate([last, pred], axis=1)                # z_t, z_hat_{t+1..t+H}
    true_full = np.concatenate([last, z_fut], axis=1)
    s_pred = predict_states(probe, pred_full[:, 1:], pred_full[:, :-1])
    s_ceil = predict_states(probe, true_full[:, 1:], true_full[:, :-1])
    s_now = predict_states(probe, z_ctx[:, -1], z_ctx[:, -2])        # state at t
    s_const = np.repeat(s_now[:, None], horizon, axis=1)

    def rmse(s):
        return np.sqrt(((s - true_states) ** 2).mean(axis=0))      # (H, 4)

    def r2(s):
        return np.stack([r2_per_variable(s[:, h], true_states[:, h]) for h in range(horizon)])

    return {
        'n': len(starts),
        'horizon': horizon,
        'skill': 1 - se / se_copy,
        'nmse': se / total_var,
        'nmse_copy': se_copy / total_var,
        'state_rmse': rmse(s_pred),
        'state_rmse_ceiling': rmse(s_ceil),
        'state_rmse_constant': rmse(s_const),
        'state_r2': r2(s_pred),
        'state_r2_ceiling': r2(s_ceil),
        'starts': starts,
        'true_states': true_states,
        'pred_states': s_pred,
        'ceiling_states': s_ceil,
        'z_pred': pred,
    }


# ----------------------------------------------------------------------------
# Test 2: action conditioning with true counterfactuals
# ----------------------------------------------------------------------------

def counterfactual_metrics(predictor, z, z_cf, cf_starts, cf_valid, cf_states,
                           probe, context_len, device, episodes=None):
    """
    Compare the predicted effect of the action with the true effect.

    For each counterfactual start t, the predictor is rolled out from the
    true context with a = 0 and with a = 1 held for H steps. The predicted
    effect is D_pred(h) = z_hat^1_{t+h} - z_hat^0_{t+h}; the true effect is
    D_true(h) = z^1_{t+h} - z^0_{t+h}, from the encoded counterfactual frames.
    Horizon h only uses the starts whose two branches are still alive at h.

    Per horizon h:
      effect_skill = 1 - mean||D_pred - D_true||^2 / mean||D_true||^2
                     (1 = perfect, 0 = same as ignoring the action)
      cosine       = mean cosine similarity between D_pred and D_true
      magnitude    = mean||D_pred|| / mean||D_true||
    and in physical units (x, x_dot, theta, theta_dot): the mean absolute
    effect from the simulator (true), from the probe on the encoded true frames
    (ceiling) and from the probe on the predictions, plus the RMSE between the
    predicted and the simulator effect.

    Returns:
        dict of per-horizon arrays (NaN where the true effect is zero, i.e. h = 1)
    """
    keep = cf_starts[:, 1] >= max(context_len, 2) - 1
    if episodes is not None:
        keep &= np.isin(cf_starts[:, 0], episodes)
    starts, z_cf, cf_states = cf_starts[keep], z_cf[keep], cf_states[keep]
    alive = cf_valid[keep].all(axis=1)                                 # (n, H)
    horizon = z_cf.shape[2]

    ep, t = starts[:, :1], starts[:, 1:]
    z_ctx = z[ep, t + np.arange(-context_len + 1, 1)]
    z_now = z[ep[:, 0], t[:, 0]]
    preds = np.stack([rollout(predictor, z_ctx, np.full((len(starts), horizon), a, np.float32), device)
                      for a in (0, 1)], axis=1)                     # (n, 2, H, z)

    def branch_states(latents):
        full = np.concatenate([np.repeat(z_now[:, None, None], 2, axis=1), latents], axis=2)
        return predict_states(probe, full[:, :, 1:], full[:, :, :-1])   # (n, 2, H, 4)

    d_true = z_cf[:, 1] - z_cf[:, 0]
    d_pred = preds[:, 1] - preds[:, 0]
    ds_sim = cf_states[:, 1] - cf_states[:, 0]
    s_pred, s_ceil = branch_states(preds), branch_states(z_cf)
    ds_pred = s_pred[:, 1] - s_pred[:, 0]
    ds_ceil = s_ceil[:, 1] - s_ceil[:, 0]

    keys = ('effect_skill', 'cosine', 'magnitude', 'latent_effect_norm')
    out = {key: np.full(horizon, np.nan) for key in keys}
    for key in ('state_effect_true', 'state_effect_ceiling', 'state_effect_pred', 'state_effect_error'):
        out[key] = np.full((horizon, 4), np.nan)
    out['n_per_h'] = alive.sum(axis=0)

    for h in range(horizon):
        m = alive[:, h]
        dt, dp = d_true[m, h], d_pred[m, h]
        nt, np_ = np.linalg.norm(dt, axis=1), np.linalg.norm(dp, axis=1)
        out['latent_effect_norm'][h] = nt.mean()
        out['state_effect_true'][h] = np.abs(ds_sim[m, h]).mean(axis=0)
        out['state_effect_ceiling'][h] = np.abs(ds_ceil[m, h]).mean(axis=0)
        out['state_effect_pred'][h] = np.abs(ds_pred[m, h]).mean(axis=0)
        out['state_effect_error'][h] = np.sqrt(((ds_pred[m, h] - ds_sim[m, h]) ** 2).mean(axis=0))
        # With Euler integration the positions (hence the frames) at h = 1 do not
        # depend on the action: the effect is undefined there
        if np.abs(ds_sim[m, h][:, [0, 2]]).max() > 1e-9:
            out['effect_skill'][h] = 1 - ((dp - dt) ** 2).sum(axis=1).mean() / (nt ** 2).mean()
            out['cosine'][h] = ((dp * dt).sum(axis=1) / np.maximum(nt * np_, 1e-12)).mean()
            out['magnitude'][h] = np_.mean() / nt.mean()

    out.update(n=len(starts), horizon=horizon, starts=starts, preds=preds, alive=alive)
    return out


def probe_ceiling(probe, z, states, dones, episodes):
    """R2 per state variable of the probe on the true latents of `episodes`."""
    preds, trues = [], []
    for ep in episodes:
        t = np.nonzero(~dones[ep, :-1])[0] + 1
        preds.append(predict_states(probe, z[ep, t], z[ep, t - 1]))
        trues.append(states[ep, t])
    return r2_per_variable(np.concatenate(preds), np.concatenate(trues))
