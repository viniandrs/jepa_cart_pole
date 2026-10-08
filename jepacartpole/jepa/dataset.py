import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from ..utils import get_data_dir


def split_episodes(num_episodes, val_fraction=0.1, seed=42):
    """
    Split episode indices into train and validation sets.

    Splitting by episode (not by frame) keeps strongly correlated consecutive
    frames from leaking between the two sets.

    Returns:
        tuple: (train_episodes, val_episodes) sorted index arrays
    """
    perm = np.random.default_rng(seed).permutation(num_episodes)
    num_val = int(round(num_episodes * val_fraction))
    return np.sort(perm[num_val:]), np.sort(perm[:num_val])


class JEPADataset(Dataset):
    """
    Transitions (z_{t-k+1..t}, a_t, z_{t+1}) from the precomputed latent
    dataset produced by data_prep.encode_dataset.

    The selected episodes are loaded into RAM. Since dones[t] = True means
    that frame t+1 comes from an environment reset, a sample (episode, t) is
    only valid if no done flag falls inside its context window, so the
    context and the target belong to the same sub-episode.
    """

    def __init__(self, h5_path=None, context_len=1, episodes=None):
        """
        Args:
            h5_path (str, optional): Latent dataset. Defaults to data/jepa/vae_latents.h5
            context_len (int): Number of past latents k in each sample
            episodes (array-like, optional): Episode indices to use (default: all)
        """
        if h5_path is None:
            h5_path = get_data_dir('jepa') / 'vae_latents.h5'
        self.h5_path = str(h5_path)
        self.context_len = context_len

        with h5py.File(self.h5_path, 'r') as h5f:
            if episodes is None:
                episodes = np.arange(h5f['mu'].shape[0])
            self.episodes = np.asarray(episodes)
            self.mu = torch.from_numpy(h5f['mu'][:][self.episodes])
            self.logvar = torch.from_numpy(h5f['logvar'][:][self.episodes])
            self.actions = torch.from_numpy(h5f['actions'][:][self.episodes]).float()
            self.dones = h5f['dones'][:][self.episodes]

        self.max_steps = self.mu.shape[1]
        self.z_dim = self.mu.shape[2]

        # Valid (episode, t): k-1 <= t <= T-2 and no done in dones[t-k+1 .. t]
        k = context_len
        steps = np.arange(k - 1, self.max_steps - 1)
        window = np.stack([self.dones[:, steps - j] for j in range(k)]).any(axis=0)
        ep_idx, step_idx = np.nonzero(~window)
        self.index = torch.from_numpy(np.stack([ep_idx, steps[step_idx]], axis=1))

    def __len__(self):
        """Return the number of valid transitions."""
        return len(self.index)

    def __getitem__(self, idx):
        """
        Returns:
            tuple: (mu_ctx, logvar_ctx, action, mu_next, logvar_next) with
                   shapes (k, z), (k, z), (1,), (z,), (z,)
        """
        ep, t = self.index[idx].tolist()
        ctx = slice(t - self.context_len + 1, t + 1)
        return (
            self.mu[ep, ctx],
            self.logvar[ep, ctx],
            self.actions[ep, t:t + 1],
            self.mu[ep, t + 1],
            self.logvar[ep, t + 1],
        )
