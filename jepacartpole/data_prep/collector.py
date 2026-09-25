"""
Parallel data collection pipeline for CartPole episodes.

This module handles efficient data collection using multiprocessing,
storing episodes in temporary HDF5 files for later consolidation.
"""

import os
import gc
import numpy as np
import h5py
from multiprocessing import Pool, cpu_count, Manager
from tqdm import tqdm
from functools import partial
from .environment import setup_pygame_headless, create_cartpole_env


def collect_episode(max_steps, episode_id, worker_id, output_dir, action_interval=20, grayscale=False):
    """
    Collect data for a single episode and write it to an HDF5 file.

    The episode continues for exactly max_steps, resetting the environment
    if the pole falls before completion.

    Args:
        max_steps (int): Maximum number of steps per episode
        episode_id (int): The episode's unique identifier
        worker_id (int): The worker's unique identifier
        output_dir (str): Directory to store episode files
        action_interval (int): Steps between random action changes
        grayscale (bool): If True, collect grayscale observations

    Returns:
        str: Path to the saved episode file
    """
    # Initialize lists to store episode data
    episode_images = []
    episode_actions = []
    episode_rewards = []
    episode_dones = []

    # Create environment
    env = create_cartpole_env(grayscale=grayscale)
    obs, _ = env.reset()

    # Generate random action for this episode
    action = np.random.randint(0, 2)

    for step in range(max_steps):
        # Store the observation
        episode_images.append(obs)

        # Change action at specified intervals
        if step % action_interval == 0:
            action = np.random.randint(0, 2)
        episode_actions.append(action)

        # Execute action and store results
        obs, reward, done, truncated, info = env.step(action)
        episode_rewards.append(reward)
        episode_dones.append(done)

        # Reset if episode terminates early
        if done:
            obs, _ = env.reset()

    # Convert lists to NumPy arrays
    episode_images = np.array(episode_images, dtype=np.uint8)
    episode_actions = np.array(episode_actions, dtype=np.float32)
    episode_rewards = np.array(episode_rewards, dtype=np.float32)
    episode_dones = np.array(episode_dones, dtype=bool)

    # Define the episode file path
    episode_filename = f'worker_{worker_id}_episode_{episode_id}.h5'
    episode_path = os.path.join(output_dir, episode_filename)

    # Write the episode data to an HDF5 file
    with h5py.File(episode_path, 'w') as h5f:
        h5f.create_dataset('images', data=episode_images, dtype='uint8')
        h5f.create_dataset('actions', data=episode_actions, dtype='float32')
        h5f.create_dataset('rewards', data=episode_rewards, dtype='float32')
        h5f.create_dataset('dones', data=episode_dones, dtype='bool')

    # Clean up memory
    env.close()
    del episode_images, episode_actions, episode_rewards, episode_dones
    del action, reward, done, truncated, info, obs, env
    gc.collect()

    return episode_path


def collect_data_chunk(args, progress_queue=None):
    """
    Worker function to collect a chunk of episodes.

    This function processes a small chunk of episodes and reports progress
    through a shared queue for real-time tracking.

    Args:
        args: Tuple containing (episode_start, episode_end, max_steps, output_dir,
                               worker_id, action_interval, grayscale)
        progress_queue: Optional multiprocessing.Queue for progress updates

    Returns:
        tuple: (worker_id, num_episodes) collected by this worker
    """
    episode_start, episode_end, max_steps, output_dir, worker_id, action_interval, grayscale = args

    num_episodes = 0
    for ep in range(episode_start, episode_end):
        collect_episode(max_steps, ep, worker_id, output_dir, action_interval, grayscale)
        num_episodes += 1

        # Report progress if queue provided
        if progress_queue is not None:
            progress_queue.put((worker_id, 1))  # (worker_id, episodes_completed)

    return (worker_id, num_episodes)


def collect_data(num_episodes=100, max_steps=500,
                output_dir=None, num_workers=None,
                action_interval=20, grayscale=False, verbose=True,
                chunk_size=None):
    """
    Collect data from CartPole environment using parallel workers.

    Each episode is stored in a separate HDF5 file for later consolidation.
    The workload is distributed across multiple workers with real-time progress
    tracking showing episodes completed by each worker.

    Args:
        num_episodes (int): Total number of episodes to generate
        max_steps (int): Maximum steps per episode
        output_dir (str, optional): Directory to store episode files.
                                   Defaults to '../data/raw'
        num_workers (int, optional): Number of worker processes.
                                    Defaults to all available CPUs
        action_interval (int): Steps between random action changes
        grayscale (bool): If True, collect grayscale observations
        verbose (bool): If True, print progress messages
        chunk_size (int, optional): Episodes per chunk. Defaults to 10 for smooth progress

    Returns:
        str: Path to the output directory containing episode files
    """
    setup_pygame_headless()

    if output_dir is None:
        output_dir = '../data/raw'

    if num_workers is None:
        num_workers = cpu_count()  # Use ALL available CPUs

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Determine chunk size for smooth progress tracking
    if chunk_size is None:
        # Smaller chunks = more frequent updates
        chunk_size = min(10, max(1, num_episodes // (num_workers * 4)))

    # Create chunks of work
    chunks = []
    episode_id = 0
    worker_id = 0

    while episode_id < num_episodes:
        chunk_end = min(episode_id + chunk_size, num_episodes)
        chunks.append((episode_id, chunk_end, max_steps, output_dir, worker_id, action_interval, grayscale))
        episode_id = chunk_end
        worker_id = (worker_id + 1) % num_workers

    num_chunks = len(chunks)

    if verbose:
        print(f"🚀 Starting parallel data collection")
        print(f"   CPUs available: {cpu_count()}")
        print(f"   Workers: {num_workers}")
        print(f"   Episodes: {num_episodes}")
        print(f"   Chunks: {num_chunks} (~{chunk_size} episodes per chunk)")
        print(f"   Steps per episode: {max_steps}")
        print(f"   Total frames: {num_episodes * max_steps:,}")
        print(f"   Output directory: {output_dir}\n")

    # Create a Manager for sharing progress between processes
    manager = Manager()
    progress_queue = manager.Queue()
    worker_counts = manager.dict({i: 0 for i in range(num_workers)})

    # Create partial function with progress_queue bound
    collect_chunk_with_queue = partial(collect_data_chunk, progress_queue=progress_queue)

    # Use Pool for multiprocessing with progress tracking
    with Pool(processes=num_workers) as pool:
        # Start async collection
        async_result = pool.map_async(collect_chunk_with_queue, chunks)

        if verbose:
            # Monitor progress in real-time
            with tqdm(total=num_episodes, desc="Collecting episodes", unit="episode") as pbar:
                # Track episodes per worker
                worker_episodes = {i: 0 for i in range(num_workers)}
                completed = 0

                # Update progress bar as episodes complete
                while completed < num_episodes:
                    if not progress_queue.empty():
                        worker_id, count = progress_queue.get()
                        worker_episodes[worker_id] += count
                        completed += count

                        # Update progress bar with worker info
                        worker_info = " | ".join([f"W{i}:{worker_episodes[i]}" for i in range(num_workers)])
                        pbar.set_postfix_str(worker_info)
                        pbar.update(count)

                    # Check if work is done
                    if async_result.ready():
                        # Drain any remaining items from queue
                        while not progress_queue.empty():
                            worker_id, count = progress_queue.get()
                            worker_episodes[worker_id] += count
                            completed += count
                            worker_info = " | ".join([f"W{i}:{worker_episodes[i]}" for i in range(num_workers)])
                            pbar.set_postfix_str(worker_info)
                            pbar.update(count)
                        break

            # Get results
            results = async_result.get()
        else:
            results = pool.map(collect_chunk_with_queue, chunks)

    if verbose:
        # Aggregate results by worker
        worker_totals = {}
        for worker_id, count in results:
            worker_totals[worker_id] = worker_totals.get(worker_id, 0) + count

        total_collected = sum(worker_totals.values())

        print(f"\n✅ Data collection completed!")
        print(f"   Total episodes collected: {total_collected}")

        # Show per-worker statistics
        print(f"   Episodes by worker:")
        for worker_id in sorted(worker_totals.keys()):
            print(f"      Worker {worker_id}: {worker_totals[worker_id]} episodes")

        print(f"   Files saved to: {output_dir}")

    return output_dir
