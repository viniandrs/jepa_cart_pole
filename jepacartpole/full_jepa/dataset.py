import h5py
import numpy as np
import torch

from ..utils import get_data_dir


def load_episodes(h5_path=None, episodes=None, keys=('images', 'actions', 'dones')):
    """
    Read the given episodes of the frame dataset into RAM, one HDF5 chunk
    (episode) at a time to avoid a second full copy in memory.

    Returns:
        dict of numpy arrays with leading dimension len(episodes)
    """
    if h5_path is None:
        h5_path = get_data_dir('vae') / 'vae_data.h5'
    with h5py.File(h5_path, 'r') as h5f:
        if episodes is None:
            episodes = np.arange(h5f['images'].shape[0])
        out = {}
        for key in keys:
            ds = h5f[key]
            out[key] = np.empty((len(episodes), *ds.shape[1:]), dtype=ds.dtype)
            for i, ep in enumerate(episodes):
                out[key][i] = ds[ep]
    return out


def frames_to_device(frames, device):
    """uint8 (..., H, W, C) numpy frames -> float (..., C, H, W) tensor in [0, 1] on device."""
    x = torch.from_numpy(frames).to(device, non_blocking=True)
    return x.movedim(-1, -3).float().div_(255.0)


class FrameWindowDataset:
    """
    Windows of frames (x_{t-k+1..t+1}, a_t) from the consolidated frame
    dataset, for training the encoder and predictor jointly.

    The selected episodes are loaded into RAM. As in JEPADataset, dones[t]
    means that frame t+1 comes from a reset, so a window (episode, t) is only
    valid if dones[t-k+1 .. t] are all False. Batches are gathered with numpy
    fancy indexing (much faster than a per-item DataLoader).
    """

    def __init__(self, h5_path=None, context_len=2, episodes=None):
        data = load_episodes(h5_path, episodes)
        self.images, self.dones = data['images'], data['dones']
        self.actions = data['actions'].astype(np.float32)
        self.context_len = context_len
        self.max_steps = self.images.shape[1]

        k = context_len
        steps = np.arange(k - 1, self.max_steps - 1)
        window = np.stack([self.dones[:, steps - j] for j in range(k)]).any(axis=0)
        ep_idx, step_idx = np.nonzero(~window)
        self.index = np.stack([ep_idx, steps[step_idx]], axis=1)

    def __len__(self):
        return len(self.index)

    def get_batch(self, idx, device):
        """
        Returns:
            tuple: frames (B, k+1, C, H, W) float in [0, 1], oldest first, the
                   last one being the target x_{t+1}; actions (B, 1)
        """
        ep, t = self.index[idx, :1], self.index[idx, 1:]
        frames = self.images[ep, t + np.arange(-self.context_len + 1, 2)]
        actions = self.actions[ep[:, 0], t[:, 0], None]
        return frames_to_device(frames, device), torch.from_numpy(actions).to(device)

    def batches(self, batch_size, device, shuffle=True, rng=None):
        """Iterate over the dataset in batches (last partial batch included)."""
        order = (rng or np.random.default_rng()).permutation(len(self)) if shuffle else np.arange(len(self))
        for s in range(0, len(order), batch_size):
            yield self.get_batch(np.sort(order[s:s + batch_size]), device)

    def num_batches(self, batch_size):
        return (len(self) + batch_size - 1) // batch_size
