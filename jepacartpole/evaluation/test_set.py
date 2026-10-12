"""
Held-out test set for world-model evaluation.

The episodes follow the same data-collection policy as data_prep.collector
(random action held for action_interval steps, environment reset when the pole
falls), with seeds disjoint from the training data. Unlike the training data,
the test set also stores the true physical state of every frame and
counterfactual futures: from a subset of states, the simulator is rolled out
with each of the two actions held constant and the resulting frames rendered.
"""

import os
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from ..data_prep.environment import (
    setup_pygame_headless, create_cartpole_env, get_observation_shape,
    CROP_TOP, CROP_BOTTOM, FRAME_SIZE
)
from ..utils import get_data_dir

STATE_NAMES = ('x', 'x_dot', 'theta', 'theta_dot')
ACTIONS = (0, 1)


def generate_test_set(output_path=None, num_episodes=100, max_steps=500,
                      action_interval=20, seed=100_000, cf_horizon=8,
                      cf_stride=5, cf_margin=4, overwrite=False, verbose=True):
    """
    Generate the held-out test set.

    Output HDF5 layout (N = num_episodes, T = max_steps, M = counterfactual starts,
    H = cf_horizon):
        images:    uint8   (N, T, 64, 64, 3)  frame before actions[t]
        actions:   float32 (N, T)
        dones:     bool    (N, T)             dones[t] => frame t+1 is a reset
        states:    float32 (N, T, 4)          (x, x_dot, theta, theta_dot) of images[t]
        cf_starts: int64   (M, 2)             (episode, t) of each counterfactual start
        cf_images: uint8   (M, 2, H, 64, 64, 3) frames t+1..t+H with action a held
        cf_states: float32 (M, 2, H, 4)
        cf_valid:  bool    (M, 2, H)          False once the counterfactual branch terminated

    Counterfactual starts are taken every cf_stride steps, at steps t with
    cf_margin <= t < T - cf_horizon and no reset in the cf_margin frames before
    t (so a predictor with context_len <= cf_margin + 1 has a valid context).

    Args:
        output_path (str, optional): Defaults to data/eval/test_set.h5
        num_episodes (int): Number of test episodes
        max_steps (int): Steps per episode
        action_interval (int): Steps between random action changes (as in training)
        seed (int): Base seed; episode i uses seed + i (disjoint from training seeds)
        cf_horizon (int): Length of each counterfactual branch
        cf_stride (int): Step interval between counterfactual starts
        cf_margin (int): Reset-free frames required before each start
        overwrite (bool): Regenerate even if the file exists
        verbose (bool): Print progress

    Returns:
        Path: Path to the test set
    """
    output_path = Path(output_path) if output_path else get_data_dir('eval') / 'test_set.h5'
    if output_path.exists() and not overwrite:
        if verbose:
            print(f"✅ Found existing test set: {output_path} (use overwrite=True to regenerate)")
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    setup_pygame_headless()
    env = create_cartpole_env()
    obs_shape = get_observation_shape()

    images = np.empty((num_episodes, max_steps, *obs_shape), dtype=np.uint8)
    actions = np.empty((num_episodes, max_steps), dtype=np.float32)
    dones = np.empty((num_episodes, max_steps), dtype=bool)
    states = np.empty((num_episodes, max_steps, 4), dtype=np.float32)

    iterator = tqdm(range(num_episodes), desc="Test episodes") if verbose else range(num_episodes)
    for ep in iterator:
        rng = np.random.default_rng(seed + ep)
        obs, _ = env.reset(seed=seed + ep)
        action = 0
        for t in range(max_steps):
            images[ep, t] = obs
            states[ep, t] = env.unwrapped.state
            if t % action_interval == 0:
                action = int(rng.integers(0, 2))
            actions[ep, t] = action
            obs, _, done, _, _ = env.step(action)
            dones[ep, t] = done
            if done:
                obs, _ = env.reset()

    # Counterfactual starts
    cf_starts = [(ep, t)
                 for ep in range(num_episodes)
                 for t in range(cf_margin, max_steps - cf_horizon, cf_stride)
                 if not dones[ep, t - cf_margin:t].any()]
    cf_starts = np.array(cf_starts, dtype=np.int64)
    num_cf = len(cf_starts)

    cf_images = np.zeros((num_cf, len(ACTIONS), cf_horizon, *obs_shape), dtype=np.uint8)
    cf_states = np.zeros((num_cf, len(ACTIONS), cf_horizon, 4), dtype=np.float32)
    cf_valid = np.zeros((num_cf, len(ACTIONS), cf_horizon), dtype=bool)

    iterator = tqdm(cf_starts, desc="Counterfactuals") if verbose else cf_starts
    for i, (ep, t) in enumerate(iterator):
        for a in ACTIONS:
            env.reset()
            env.unwrapped.state = states[ep, t].astype(np.float64)
            for h in range(cf_horizon):
                obs, _, done, _, _ = env.step(a)
                cf_images[i, a, h] = obs
                cf_states[i, a, h] = env.unwrapped.state
                cf_valid[i, a, h] = True
                if done:
                    break

    env.close()

    with h5py.File(output_path, 'w') as h5f:
        h5f.create_dataset('images', data=images, chunks=(1, max_steps, *obs_shape))
        h5f.create_dataset('actions', data=actions)
        h5f.create_dataset('dones', data=dones)
        h5f.create_dataset('states', data=states)
        h5f.create_dataset('cf_starts', data=cf_starts)
        h5f.create_dataset('cf_images', data=cf_images, chunks=(1, len(ACTIONS), cf_horizon, *obs_shape))
        h5f.create_dataset('cf_states', data=cf_states)
        h5f.create_dataset('cf_valid', data=cf_valid)
        h5f.attrs.update(seed=seed, action_interval=action_interval, cf_horizon=cf_horizon,
                         cf_stride=cf_stride, cf_margin=cf_margin, state_names=list(STATE_NAMES),
                         crop_rows=(CROP_TOP, CROP_BOTTOM), frame_size=FRAME_SIZE)

    if verbose:
        size_mb = os.path.getsize(output_path) / 1024 ** 2
        print(f"Test set saved to: {output_path} ({size_mb:.0f} MB)")
        print(f"  {num_episodes} episodes × {max_steps} steps, {dones.sum()} resets")
        print(f"  {num_cf} counterfactual starts × {len(ACTIONS)} actions × {cf_horizon} steps "
              f"({cf_valid.all(axis=(1, 2)).mean():.0%} with both branches alive for all steps)")

    return output_path


def load_test_set(path=None):
    """
    Load the whole test set into memory.

    Returns:
        dict: arrays keyed as in the HDF5 file, plus 'attrs'
    """
    path = path or get_data_dir('eval') / 'test_set.h5'
    with h5py.File(path, 'r') as h5f:
        data = {key: h5f[key][:] for key in h5f.keys()}
        data['attrs'] = dict(h5f.attrs)
    return data
