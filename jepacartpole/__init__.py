"""
JEPA CartPole: Comparing VAE and Transfer Learning Encoders for World Models

This package implements a JEPA world model for the CartPole-v1 Gymnasium environment,
comparing two encoding strategies: a trained VAE encoder and a pretrained transfer
learning encoder.

Import submodules explicitly:
    from jepacartpole import config
    from jepacartpole.data_prep import collect_data
    from jepacartpole.vae import VAE
    etc.
"""

__version__ = '0.1.0'

# Submodules are available for explicit import but not automatically loaded
__all__ = [
    'config',
    'utils',
    'data_prep',
    'vae',
    'jepa',
    'transfer_learning',
    'rl',
]
