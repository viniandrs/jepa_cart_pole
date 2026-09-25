"""
Environment creation utilities for CartPole data collection.
"""

import os
import gymnasium as gym
from gymnasium.wrappers import AddRenderObservation, TransformObservation
import numpy as np


def setup_pygame_headless():
    """
    Configure Pygame to run in headless mode (no display required).
    Call this before creating any environments.
    """
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
    os.environ['SDL_AUDIODRIVER'] = 'dummy'
    os.environ['SDL_VIDEODRIVER'] = 'dummy'


def create_cartpole_env(grayscale=False):
    """
    Create a CartPole-v1 environment with pixel observations.

    Args:
        grayscale (bool): If True, convert observations to grayscale

    Returns:
        gym.Env: Configured CartPole environment with RGB/grayscale pixel observations
    """
    env = gym.make('CartPole-v1', render_mode="rgb_array")
    env = AddRenderObservation(env, render_only=True)

    if grayscale:
        # Convert RGB to grayscale by averaging across channels
        env = TransformObservation(
            env,
            lambda obs: np.mean(obs, axis=2, keepdims=True),
            env.observation_space
        )

    return env


def get_observation_shape(grayscale=False):
    """
    Get the observation shape for CartPole environment.

    Args:
        grayscale (bool): If True, returns grayscale shape

    Returns:
        tuple: Observation shape (height, width, channels)
    """
    if grayscale:
        return (400, 600, 1)
    else:
        return (400, 600, 3)
