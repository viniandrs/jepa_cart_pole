"""
Parallel data collection pipeline for CartPole episodes.

Following World Models (Ha & Schmidhuber, 2018), each episode is stored as a
compressed NumPy file (.npz) with preprocessed 64x64 frames. The episode files
are later consolidated into a single HDF5 dataset by merge.py.
"""

import os
import numpy as np
from multiprocessing import Pool, cpu_count, Manager
from tqdm import tqdm
from functools import partial
from .environment import setup_pygame_headless, create_cartpole_env, get_observation_shape
from ..utils import get_data_dir


def episode_filename(episode_id):
    """Return the file name used to store a given episode."""
    return f'episode_{episode_id:05d}.npz'


def collect_episode(max_steps, episode_id, output_dir, action_interval=20,
                    grayscale=False, seed=42):
    """
    Collect data for a single episode and write it to a compressed .npz file.

    The episode continues for exactly max_steps, resetting the environment
    if the pole falls before completion.

    Args:
        max_steps (int): Maximum number of steps per episode
        episode_id (int): The episode's unique identifier
        output_dir (str): Directory to store episode files
        action_interval (int): Steps between random action changes
        grayscale (bool): If True, collect grayscale observations
        seed (int): Base seed; the episode uses seed + episode_id

    Returns:
        str: Path to the saved episode file
    """
    episode_path = os.path.join(output_dir, episode_filename(episode_id))

    # Seed per episode so forked workers don't share the same random stream
    rng = np.random.default_rng(seed + episode_id)

    episode_images = np.empty((max_steps, *get_observation_shape(grayscale)), dtype=np.uint8)
    episode_actions = np.empty(max_steps, dtype=np.float32)
    episode_rewards = np.empty(max_steps, dtype=np.float32)
    episode_dones = np.empty(max_steps, dtype=bool)

    env = create_cartpole_env(grayscale=grayscale)
    obs, _ = env.reset(seed=seed + episode_id)
    action = 0

    for step in range(max_steps):
        episode_images[step] = obs

        # Change action at specified intervals
        if step % action_interval == 0:
            action = int(rng.integers(0, 2))
        episode_actions[step] = action

        obs, reward, done, truncated, info = env.step(action)
        episode_rewards[step] = reward
        episode_dones[step] = done

        # Reset if episode terminates early
        if done:
            obs, _ = env.reset()

    env.close()

    np.savez_compressed(
        episode_path,
        images=episode_images,
        actions=episode_actions,
        rewards=episode_rewards,
        dones=episode_dones
    )

    return episode_path


def collect_data_chunk(args, progress_queue=None):
    """
    Worker function to collect a chunk of episodes.

    This function processes a small chunk of episodes and reports progress
    through a shared queue for real-time tracking.

    Args:
        args: Tuple containing (episode_ids, max_steps, output_dir,
                               worker_id, action_interval, grayscale, seed)
        progress_queue: Optional multiprocessing.Queue for progress updates

    Returns:
        tuple: (worker_id, num_episodes) collected by this worker
    """
    episode_ids, max_steps, output_dir, worker_id, action_interval, grayscale, seed = args

    num_episodes = 0
    for ep in episode_ids:
        collect_episode(max_steps, ep, output_dir, action_interval, grayscale, seed)
        num_episodes += 1

        # Report progress if queue provided
        if progress_queue is not None:
            progress_queue.put((worker_id, 1))  # (worker_id, episodes_completed)

    return (worker_id, num_episodes)


def collect_data(num_episodes=100, max_steps=500,
                output_dir=None, num_workers=None,
                action_interval=20, grayscale=False, verbose=True,
                chunk_size=None, seed=42):
    """
    Collect data from CartPole environment using parallel workers.

    Each episode is stored in a separate .npz file for later consolidation.
    Episodes whose file already exists are skipped, so an interrupted
    collection can be resumed by calling this function again.

    Args:
        num_episodes (int): Total number of episodes to generate
        max_steps (int): Maximum steps per episode
        output_dir (str, optional): Directory to store episode files.
                                   Defaults to data/raw
        num_workers (int, optional): Number of worker processes.
                                    Defaults to all available CPUs
        action_interval (int): Steps between random action changes
        grayscale (bool): If True, collect grayscale observations
        verbose (bool): If True, print progress messages
        chunk_size (int, optional): Episodes per chunk. Defaults to 10 for smooth progress
        seed (int): Base random seed (episode i uses seed + i)

    Returns:
        str: Path to the output directory
    """
    setup_pygame_headless()

    if output_dir is None:
        output_dir = get_data_dir('raw')
    output_dir = str(output_dir)

    if num_workers is None:
        num_workers = cpu_count()  # Use ALL available CPUs

    os.makedirs(output_dir, exist_ok=True)

    # Skip episodes that were already collected
    pending = [
        ep for ep in range(num_episodes)
        if not os.path.exists(os.path.join(output_dir, episode_filename(ep)))
    ]
    num_pending = len(pending)

    if num_pending == 0:
        if verbose:
            print(f"⏭️  All {num_episodes} episodes already exist in {output_dir}")
        return output_dir

    # Determine chunk size for smooth progress tracking
    if chunk_size is None:
        # Smaller chunks = more frequent updates
        chunk_size = min(10, max(1, num_pending // (num_workers * 4)))

    # Create chunks of work
    chunks = []
    for i, start in enumerate(range(0, num_pending, chunk_size)):
        episode_ids = pending[start:start + chunk_size]
        worker_id = i % num_workers
        chunks.append((episode_ids, max_steps, output_dir, worker_id, action_interval, grayscale, seed))

    if verbose:
        print(f"🚀 Starting parallel data collection")
        print(f"   CPUs available: {cpu_count()}")
        print(f"   Workers: {num_workers}")
        print(f"   Episodes: {num_episodes} ({num_episodes - num_pending} already collected)")
        print(f"   Chunks: {len(chunks)} (~{chunk_size} episodes per chunk)")
        print(f"   Steps per episode: {max_steps}")
        print(f"   Total frames: {num_episodes * max_steps:,}")
        print(f"   Output directory: {output_dir}")
        print()

    # Create a Manager for sharing progress between processes
    manager = Manager()
    progress_queue = manager.Queue()

    # Create partial function with progress_queue bound
    collect_chunk_with_queue = partial(collect_data_chunk, progress_queue=progress_queue)

    # Use Pool for multiprocessing with progress tracking
    with Pool(processes=num_workers) as pool:
        async_result = pool.map_async(collect_chunk_with_queue, chunks)

        if verbose:
            # Monitor progress in real-time
            with tqdm(total=num_pending, desc="Collecting episodes", unit="episode") as pbar:
                worker_episodes = {i: 0 for i in range(num_workers)}
                completed = 0

                while completed < num_pending:
                    # Block briefly instead of busy-waiting on the queue
                    try:
                        worker_id, count = progress_queue.get(timeout=1)
                    except Exception:
                        if async_result.ready():
                            break
                        continue

                    worker_episodes[worker_id] += count
                    completed += count

                    worker_info = " | ".join([f"W{i}:{worker_episodes[i]}" for i in range(num_workers)])
                    pbar.set_postfix_str(worker_info)
                    pbar.update(count)

        # Re-raises any exception from the workers
        results = async_result.get()

    if verbose:
        total_collected = sum(count for _, count in results)
        print(f"\n✅ Data collection completed!")
        print(f"   Episodes collected in this run: {total_collected}")
        print(f"   Files saved to: {output_dir}")

    return output_dir
