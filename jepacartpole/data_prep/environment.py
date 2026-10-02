"""
Environment creation utilities for CartPole data collection.

Frames are preprocessed following World Models (Ha & Schmidhuber, 2018):
the raw render is cropped to the region where the cart and pole can appear
and resized to 64x64 uint8.
"""

import os
import cv2
import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.wrappers import AddRenderObservation, TransformObservation
import numpy as np

# Raw render size of CartPole-v1 (height, width)
RAW_HEIGHT, RAW_WIDTH = 400, 600

# Rows kept from the raw render. The cart spans rows ~285-315 and the pole
# reaches up to row ~172 (measured over random rollouts), so this band keeps
# all non-background pixels with some margin. Full width is kept because the
# cart can move across the whole screen.
CROP_TOP, CROP_BOTTOM = 150, 330

# Side of the square preprocessed frame
FRAME_SIZE = 64


def setup_pygame_headless():
    """
    Configure Pygame to run in headless mode (no display required).
    Call this before creating any environments.
    """
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
    os.environ['SDL_AUDIODRIVER'] = 'dummy'
    os.environ['SDL_VIDEODRIVER'] = 'dummy'


def preprocess_frame(frame, grayscale=False):
    """
    Crop a raw CartPole render and resize it to FRAME_SIZE x FRAME_SIZE.

    Args:
        frame (np.ndarray): Raw RGB frame of shape (400, 600, 3), uint8
        grayscale (bool): If True, average the channels into a single one

    Returns:
        np.ndarray: Preprocessed frame of shape (64, 64, C), uint8
    """
    frame = frame[CROP_TOP:CROP_BOTTOM]
    frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)

    if grayscale:
        frame = frame.mean(axis=2, keepdims=True).round().astype(np.uint8)

    return frame


def create_cartpole_env(grayscale=False, preprocess=True):
    """
    Create a CartPole-v1 environment with pixel observations.

    Args:
        grayscale (bool): If True, convert observations to grayscale
        preprocess (bool): If True, crop and resize observations to 64x64.
                           If False, return raw 400x600 renders

    Returns:
        gym.Env: Configured CartPole environment with RGB/grayscale pixel observations
    """
    env = gym.make('CartPole-v1', render_mode="rgb_array")
    env = AddRenderObservation(env, render_only=True)

    if preprocess:
        env = TransformObservation(
            env,
            lambda obs: preprocess_frame(obs, grayscale=grayscale),
            Box(0, 255, get_observation_shape(grayscale), dtype=np.uint8)
        )
    elif grayscale:
        # Convert RGB to grayscale by averaging across channels
        env = TransformObservation(
            env,
            lambda obs: obs.mean(axis=2, keepdims=True).round().astype(np.uint8),
            Box(0, 255, (RAW_HEIGHT, RAW_WIDTH, 1), dtype=np.uint8)
        )

    return env


def get_observation_shape(grayscale=False):
    """
    Get the preprocessed observation shape for CartPole environment.

    Args:
        grayscale (bool): If True, returns grayscale shape

    Returns:
        tuple: Observation shape (height, width, channels)
    """
    return (FRAME_SIZE, FRAME_SIZE, 1 if grayscale else 3)
