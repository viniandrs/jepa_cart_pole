"""
Encoder-agnostic evaluation protocol for latent world models.

Used by notebooks/*/4-*_evaluation.ipynb so that world models built on
different encoders are evaluated on the same held-out test set with the
same metrics.
"""

from .test_set import generate_test_set, load_test_set, STATE_NAMES
from .models import load_world_model, encode_frames
from .probes import fit_state_probe, predict_states, r2_per_variable
from .metrics import (
    valid_starts,
    gather_segments,
    rollout,
    encoder_diagnostics,
    one_step_metrics,
    rollout_metrics,
    counterfactual_metrics,
    probe_ceiling
)
from .plots import (
    plot_state_distribution,
    plot_encoder_diagnostics,
    plot_one_step,
    plot_latent_rollout,
    plot_state_rollout,
    plot_state_trajectories,
    plot_counterfactual,
    plot_decoded_rollout,
    plot_decoded_counterfactual
)
from .scorecard import build_scorecard, save_scorecard, load_scorecards

__all__ = [
    'generate_test_set', 'load_test_set', 'STATE_NAMES',
    'load_world_model', 'encode_frames',
    'fit_state_probe', 'predict_states', 'r2_per_variable',
    'valid_starts', 'gather_segments', 'rollout',
    'encoder_diagnostics', 'one_step_metrics', 'rollout_metrics',
    'counterfactual_metrics', 'probe_ceiling',
    'plot_state_distribution', 'plot_encoder_diagnostics', 'plot_one_step',
    'plot_latent_rollout', 'plot_state_rollout', 'plot_state_trajectories',
    'plot_counterfactual', 'plot_decoded_rollout', 'plot_decoded_counterfactual',
    'build_scorecard', 'save_scorecard', 'load_scorecards',
]
