"""
Parallel data collection pipeline for CartPole episodes.

This module handles efficient data collection using multiprocessing,
storing episodes in temporary HDF5 files with S3 support to avoid
local disk space issues.
"""

import os
import gc
import glob
import numpy as np
import h5py
from multiprocessing import Pool, cpu_count, Manager
from tqdm import tqdm
from functools import partial
from pathlib import Path
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


def upload_batch_to_s3(local_dir, s3_manager, s3_prefix, verbose=True):
    """
    Upload all HDF5 files in local directory to S3 and delete them locally.

    Args:
        local_dir (str): Local directory containing HDF5 files
        s3_manager: S3Manager instance
        s3_prefix (str): S3 prefix for uploaded files
        verbose (bool): Print progress messages

    Returns:
        int: Number of files uploaded
    """
    # Find all HDF5 files in local directory
    local_files = glob.glob(os.path.join(local_dir, '*.h5'))

    if not local_files:
        return 0

    # Upload files and delete locally
    uploaded = s3_manager.upload_files(
        local_files,
        s3_prefix,
        delete_after_upload=True
    )

    return uploaded


def collect_data(num_episodes=100, max_steps=500,
                output_dir=None, num_workers=None,
                action_interval=20, grayscale=False, verbose=True,
                chunk_size=None, use_s3=False, s3_bucket=None,
                s3_prefix=None, batch_size=50):
    """
    Collect data from CartPole environment using parallel workers.

    Supports saving to S3 in batches to avoid filling local disk space.
    Each episode is stored in a separate HDF5 file for later consolidation.

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
        use_s3 (bool): If True, upload files to S3 in batches during collection
        s3_bucket (str): S3 bucket name (required if use_s3=True)
        s3_prefix (str): S3 prefix/directory (required if use_s3=True)
        batch_size (int): Number of episodes to collect before uploading to S3

    Returns:
        str: Path to the output directory (local) or S3 prefix
    """
    setup_pygame_headless()

    if output_dir is None:
        output_dir = '../data/raw'

    if num_workers is None:
        num_workers = cpu_count()  # Use ALL available CPUs

    # Validate S3 parameters
    if use_s3:
        if not s3_bucket or not s3_prefix:
            raise ValueError("s3_bucket and s3_prefix are required when use_s3=True")

        # Import S3Manager
        from jepacartpole.storage import S3Manager
        s3_manager = S3Manager(s3_bucket, verbose=verbose)

        # Check if files already exist in S3
        existing_files = s3_manager.list_files(s3_prefix)
        if existing_files:
            if verbose:
                print(f"⚠️  Found {len(existing_files)} existing files in S3")
                print(f"   s3://{s3_bucket}/{s3_prefix}")
                print(f"   Skipping data collection - files already exist!")
            return s3_prefix

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
        if use_s3:
            print(f"   Storage: S3 (batch size: {batch_size} episodes)")
            print(f"   S3 Bucket: {s3_bucket}")
            print(f"   S3 Prefix: {s3_prefix}")
        else:
            print(f"   Storage: Local disk")
            print(f"   Output directory: {output_dir}")
        print()

    # Create a Manager for sharing progress between processes
    manager = Manager()
    progress_queue = manager.Queue()

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
                last_batch_upload = 0

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

                        # Upload batch to S3 if threshold reached
                        if use_s3 and (completed - last_batch_upload) >= batch_size:
                            if verbose:
                                pbar.write(f"\n📤 Uploading batch to S3 ({completed} episodes collected)...")

                            uploaded = upload_batch_to_s3(output_dir, s3_manager, s3_prefix, verbose=False)

                            if verbose:
                                pbar.write(f"✅ Uploaded {uploaded} files, freed local disk space\n")

                            last_batch_upload = completed

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

    # Upload any remaining files to S3
    if use_s3:
        if verbose:
            print(f"\n📤 Uploading final batch to S3...")

        remaining_files = glob.glob(os.path.join(output_dir, '*.h5'))
        if remaining_files:
            uploaded = upload_batch_to_s3(output_dir, s3_manager, s3_prefix, verbose=verbose)
            if verbose:
                print(f"✅ Uploaded {uploaded} files")

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

        if use_s3:
            print(f"   Files saved to: s3://{s3_bucket}/{s3_prefix}")
        else:
            print(f"   Files saved to: {output_dir}")

    return s3_prefix if use_s3 else output_dir
