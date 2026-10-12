"""
Utility functions for the JEPA CartPole project.

This module contains quick access routines and helper functions used across
the project stages.
"""

import torch
import numpy as np
from pathlib import Path


def get_device():
    """
    Get the appropriate device (CUDA if available, otherwise CPU).

    Returns:
        torch.device: The device to use for computations
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_project_root():
    """
    Get the project root directory.

    Returns:
        Path: Path to the project root directory
    """
    return Path(__file__).parent.parent


def get_data_dir(stage=None):
    """
    Get the data directory path for a specific stage.

    Args:
        stage (str, optional): Stage name ('vae', 'jepa', 'raw', etc.)
                              If None, returns the base data directory

    Returns:
        Path: Path to the data directory
    """
    data_dir = get_project_root() / 'data'
    if stage is not None:
        data_dir = data_dir / stage
    return data_dir


def get_checkpoint_dir():
    """
    Get the checkpoints directory path.

    Returns:
        Path: Path to the checkpoints directory
    """
    return get_project_root() / 'checkpoints'


def get_results_dir(stage=None):
    """
    Get the results directory path for a specific stage.

    Args:
        stage (str, optional): Stage name ('data_prep', 'vae', 'transfer_learning', 'rl')
                              If None, returns the base results directory

    Returns:
        Path: Path to the results directory
    """
    results_dir = get_project_root() / 'results'
    if stage is not None:
        results_dir = results_dir / stage
    return results_dir


def save_checkpoint(model, path, **kwargs):
    """
    Save a model checkpoint with additional metadata.

    Args:
        model (torch.nn.Module): The model to save
        path (str or Path): Path to save the checkpoint
        **kwargs: Additional metadata to save (e.g., epoch, losses, config)
    """
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        **kwargs
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(model, path, device=None):
    """
    Load a model checkpoint.

    Args:
        model (torch.nn.Module): The model to load weights into
        path (str or Path): Path to the checkpoint file
        device (torch.device, optional): Device to map the model to

    Returns:
        dict: The full checkpoint dictionary (including metadata)
    """
    if device is None:
        device = get_device()

    checkpoint = torch.load(path, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    return checkpoint


def set_seed(seed=42):
    """
    Set random seeds for reproducibility.

    Args:
        seed (int): Random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.

    Args:
        model (torch.nn.Module): The model

    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
