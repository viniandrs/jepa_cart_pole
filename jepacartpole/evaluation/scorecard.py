"""
Scorecard: a flat summary of one world-model evaluation, saved as JSON so
that runs with different encoders can be compared side by side.
"""

import json
from datetime import date

import numpy as np

from ..utils import get_results_dir
from .test_set import STATE_NAMES

REPORT_HORIZONS = (1, 4, 8)


def build_scorecard(run_name, encoder_type, diag, ceiling, one_step, roll, cf, one_step_state=None,
                    probe=None, notes=''):
    """
    Collect the headline numbers of an evaluation run.

    Returns:
        dict of JSON-serializable scalars
    """
    card = {
        'run_name': run_name,
        'encoder_type': encoder_type,
        'date': date.today().isoformat(),
        'notes': notes,
        'latent_dim': diag['latent_dim'],
        'effective_dim': round(diag['effective_dim'], 3),
        'active_dims': diag.get('active_dims'),
        'probe_components': int(probe[0].n_components_) if probe is not None else None,
        'one_step_skill': round(one_step['skill'], 4),
        'one_step_nmse': round(one_step['nmse'], 5),
    }
    for name, r2 in zip(STATE_NAMES, ceiling):
        card[f'probe_ceiling_r2_{name}'] = round(float(r2), 4)
    if one_step_state is not None:   # rollout_metrics with horizon=1 (all transitions)
        for i, name in enumerate(STATE_NAMES):
            card[f'one_step_state_rmse_{name}'] = round(float(one_step_state['state_rmse'][0, i]), 4)
            card[f'one_step_state_rmse_{name}_ceiling'] = round(float(one_step_state['state_rmse_ceiling'][0, i]), 4)
            card[f'one_step_state_rmse_{name}_constant'] = round(float(one_step_state['state_rmse_constant'][0, i]), 4)
    for h in REPORT_HORIZONS:
        if h <= roll['horizon']:
            card[f'rollout_skill_h{h}'] = round(float(roll['skill'][h - 1]), 4)
            for i, name in enumerate(STATE_NAMES):
                card[f'state_rmse_{name}_h{h}'] = round(float(roll['state_rmse'][h - 1, i]), 4)
                card[f'state_rmse_{name}_h{h}_constant'] = round(float(roll['state_rmse_constant'][h - 1, i]), 4)
        if h <= cf['horizon'] and not np.isnan(cf['effect_skill'][h - 1]):
            card[f'action_effect_skill_h{h}'] = round(float(cf['effect_skill'][h - 1]), 4)
            card[f'action_effect_cosine_h{h}'] = round(float(cf['cosine'][h - 1]), 4)
    return card


def save_scorecard(card, results_dir=None):
    """Write the scorecard to results/eval/<run_name>.json and return the path."""
    out_dir = results_dir or get_results_dir('eval')
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{card['run_name']}.json"
    path.write_text(json.dumps(card, indent=2))
    return path


def load_scorecards(results_dir=None):
    """
    Load every saved scorecard as a pandas DataFrame (one column per run,
    one row per metric), for side-by-side comparison.
    """
    import pandas as pd
    out_dir = results_dir or get_results_dir('eval')
    cards = [json.loads(p.read_text()) for p in sorted(out_dir.glob('*.json'))]
    if not cards:
        return pd.DataFrame()
    return pd.DataFrame(cards).set_index('run_name').T
