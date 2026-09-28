"""
Chunked consolidation of HDF5 files from S3.

This module consolidates episode files from S3 into multiple chunk files,
avoiding the seek() limitation of S3 while minimizing local disk usage.

Strategy:
1. Process episodes in chunks (e.g., 50 episodes per chunk)
2. Write each chunk locally (uses ~7-8GB for 50 episodes)
3. Upload chunk to S3 and delete locally
4. Repeat until all episodes processed
"""

import os
import gc
import h5py
import s3fs
import boto3
import shutil
from tqdm import tqdm
from pathlib import Path


def consolidate_episodes_in_chunks(
    s3_bucket: str,
    s3_prefix: str,
    s3_output_prefix: str,
    chunk_size: int = 50,
    max_steps: int = 500,
    local_temp_dir: str = '/tmp/consolidation',
    verify_chunks: bool = True,
    delete_local_after_upload: bool = True,
    skip_if_exists: bool = True,
    verbose: bool = True
):
    """
    Consolidate HDF5 episode files from S3 into multiple chunk files.

    This approach avoids the seek() limitation when writing HDF5 to S3 by:
    1. Writing chunks locally (50 episodes = ~7-8GB)
    2. Uploading each chunk to S3
    3. Deleting local chunk to free space
    4. Repeating for all episodes

    Args:
        s3_bucket (str): S3 bucket name
        s3_prefix (str): S3 prefix containing source episode files
        s3_output_prefix (str): S3 prefix for output chunks (e.g., 'chunks/dataset')
        chunk_size (int): Number of episodes per chunk (default: 50)
        max_steps (int): Steps per episode
        local_temp_dir (str): Local directory for temporary chunk files
        verify_chunks (bool): Verify each chunk after creation
        delete_local_after_upload (bool): Delete local chunk after S3 upload
        skip_if_exists (bool): Skip consolidation if chunks already exist in S3 (default: True)
        verbose (bool): Print progress messages

    Returns:
        dict: Statistics about consolidation (num_chunks, total_episodes, etc.)
                If skipped, includes 'skipped': True in the result
    """
    if verbose:
        print("=" * 80)
        print("HDF5 CHUNKED CONSOLIDATION")
        print("=" * 80)
        print(f"Source: s3://{s3_bucket}/{s3_prefix}")
        print(f"Destination: s3://{s3_bucket}/{s3_output_prefix}/chunk_*.h5")
        print(f"Chunk size: {chunk_size} episodes")
        print(f"Local temp: {local_temp_dir}")
        print()

    # Initialize S3
    fs = s3fs.S3FileSystem()
    s3_client = boto3.client('s3')

    # Step 0: Check if chunks already exist
    if skip_if_exists:
        if verbose:
            print("🔍 Checking if chunks already exist in S3...")

        s3_output_path = f"{s3_bucket}/{s3_output_prefix}"
        try:
            existing_files = fs.ls(s3_output_path)
            existing_chunks = [f for f in existing_files if f.endswith('.h5') and 'chunk_' in f]
            existing_chunks.sort()

            if existing_chunks:
                if verbose:
                    print(f"✓ Found {len(existing_chunks)} existing chunks in S3")
                    print()
                    print("=" * 80)
                    print("⏭️  SKIPPING CONSOLIDATION - CHUNKS ALREADY EXIST")
                    print("=" * 80)
                    print(f"Location: s3://{s3_output_path}/")
                    print(f"Existing chunks: {len(existing_chunks)}")
                    print()

                    # Get info from existing chunks
                    total_episodes = 0
                    chunk_info = []

                    for chunk_file in existing_chunks:
                        try:
                            with fs.open(chunk_file, 'rb') as f:
                                with h5py.File(f, 'r') as h5f:
                                    num_episodes = h5f['images'].shape[0]
                                    total_episodes += num_episodes

                            # Get file size
                            file_info = fs.info(chunk_file)
                            size_mb = file_info['size'] / (1024 * 1024)

                            chunk_idx = int(chunk_file.split('chunk_')[1].split('.')[0])
                            chunk_info.append({
                                'chunk_idx': chunk_idx,
                                's3_key': chunk_file,
                                'num_episodes': num_episodes,
                                'size_mb': size_mb
                            })

                        except Exception as e:
                            if verbose:
                                print(f"⚠️  Warning: Could not read {chunk_file}: {e}")

                    if verbose:
                        print(f"Total episodes in existing chunks: {total_episodes}")
                        print()
                        print("To force re-consolidation, set skip_if_exists=False")
                        print("=" * 80)

                return {
                    'num_chunks': len(existing_chunks),
                    'total_episodes': total_episodes,
                    'chunk_size': chunk_size,
                    'chunks': chunk_info,
                    's3_location': f"s3://{s3_bucket}/{s3_output_prefix}/",
                    'skipped': True
                }
        except FileNotFoundError:
            if verbose:
                print(f"✓ No existing chunks found - will create new ones")
                print()
        except Exception as e:
            if verbose:
                print(f"⚠️  Warning: Could not check for existing chunks: {e}")
                print("   Proceeding with consolidation...")
                print()

    # Create temp directory
    os.makedirs(local_temp_dir, exist_ok=True)

    # Step 1: List all episode files in S3
    if verbose:
        print("📋 Step 1: Listing source files in S3...")

    s3_path = f"{s3_bucket}/{s3_prefix}"
    try:
        all_files = fs.ls(s3_path)
        episode_files = [f for f in all_files if f.endswith('.h5') or f.endswith('.hdf5')]
        episode_files.sort()
    except Exception as e:
        raise ValueError(f"Failed to list files in S3: {e}")

    num_episodes = len(episode_files)
    if num_episodes == 0:
        raise ValueError(f"No HDF5 files found in s3://{s3_path}")

    if verbose:
        print(f"✓ Found {num_episodes} episode files")
        print()

    # Step 2: Determine structure from first file
    if verbose:
        print("🔍 Step 2: Analyzing first file...")

    try:
        with fs.open(episode_files[0], 'rb') as f:
            with h5py.File(f, 'r') as h5f:
                if 'images' not in h5f:
                    raise ValueError("First file missing 'images' dataset")

                image_shape = h5f['images'].shape
                if image_shape[0] != max_steps:
                    raise ValueError(f"First file has {image_shape[0]} steps, expected {max_steps}")

                image_dims = image_shape[1:]  # (height, width, channels)

                required = ['images', 'actions', 'rewards', 'dones']
                missing = [ds for ds in required if ds not in h5f]
                if missing:
                    raise ValueError(f"First file missing datasets: {missing}")

    except Exception as e:
        raise RuntimeError(f"Failed to analyze first file: {e}")

    if verbose:
        print(f"✓ Image shape: {image_dims}")
        print(f"✓ Steps per episode: {max_steps}")
        print()

    # Step 3: Process episodes in chunks
    if verbose:
        print("🔄 Step 3: Processing episodes in chunks...")
        print()

    num_chunks = (num_episodes + chunk_size - 1) // chunk_size
    chunk_info = []
    successful_episodes = 0

    for chunk_idx in range(num_chunks):
        start_ep = chunk_idx * chunk_size
        end_ep = min(start_ep + chunk_size, num_episodes)
        chunk_episodes = episode_files[start_ep:end_ep]
        actual_chunk_size = len(chunk_episodes)

        if verbose:
            print(f"📦 Processing chunk {chunk_idx + 1}/{num_chunks}")
            print(f"   Episodes: {start_ep} to {end_ep - 1} ({actual_chunk_size} episodes)")

        # Local chunk file path
        chunk_filename = f"chunk_{chunk_idx:04d}.h5"
        local_chunk_path = os.path.join(local_temp_dir, chunk_filename)

        # Create chunk locally
        try:
            if verbose:
                print(f"   Creating local chunk: {local_chunk_path}")

            with h5py.File(local_chunk_path, 'w') as chunk_h5:
                # Create datasets for this chunk
                images_shape = (actual_chunk_size, max_steps, *image_dims)
                actions_shape = (actual_chunk_size, max_steps)
                rewards_shape = (actual_chunk_size, max_steps)
                dones_shape = (actual_chunk_size, max_steps)

                chunk_h5.create_dataset(
                    'images',
                    shape=images_shape,
                    dtype='uint8',
                    chunks=(1, max_steps, *image_dims)
                )
                chunk_h5.create_dataset(
                    'actions',
                    shape=actions_shape,
                    dtype='float32',
                    chunks=(1, max_steps)
                )
                chunk_h5.create_dataset(
                    'rewards',
                    shape=rewards_shape,
                    dtype='float32',
                    chunks=(1, max_steps)
                )
                chunk_h5.create_dataset(
                    'dones',
                    shape=dones_shape,
                    dtype='bool',
                    chunks=(1, max_steps)
                )

                # Fill chunk with episodes
                pbar = tqdm(
                    enumerate(chunk_episodes),
                    total=actual_chunk_size,
                    desc=f"   Chunk {chunk_idx + 1}",
                    unit="ep",
                    leave=False
                ) if verbose else enumerate(chunk_episodes)

                for local_idx, s3_file in pbar:
                    try:
                        with fs.open(s3_file, 'rb') as f:
                            with h5py.File(f, 'r') as source:
                                chunk_h5['images'][local_idx] = source['images'][:]
                                chunk_h5['actions'][local_idx] = source['actions'][:]
                                chunk_h5['rewards'][local_idx] = source['rewards'][:]
                                chunk_h5['dones'][local_idx] = source['dones'][:]

                        successful_episodes += 1
                        gc.collect()

                    except Exception as e:
                        if verbose:
                            print(f"\n   ⚠️  Failed to process {s3_file}: {e}")

            # Verify chunk if requested
            if verify_chunks:
                if verbose:
                    print(f"   ✓ Verifying chunk...")

                with h5py.File(local_chunk_path, 'r') as chunk_h5:
                    assert chunk_h5['images'].shape[0] == actual_chunk_size
                    assert 'actions' in chunk_h5
                    assert 'rewards' in chunk_h5
                    assert 'dones' in chunk_h5

            # Get chunk size
            chunk_size_mb = os.path.getsize(local_chunk_path) / (1024 * 1024)

            if verbose:
                print(f"   ✓ Local chunk created ({chunk_size_mb:.2f} MB)")

            # Upload to S3
            s3_chunk_key = f"{s3_output_prefix}/{chunk_filename}"

            if verbose:
                print(f"   📤 Uploading to s3://{s3_bucket}/{s3_chunk_key}...")

            s3_client.upload_file(local_chunk_path, s3_bucket, s3_chunk_key)

            if verbose:
                print(f"   ✓ Uploaded to S3")

            # Store chunk info
            chunk_info.append({
                'chunk_idx': chunk_idx,
                's3_key': s3_chunk_key,
                'num_episodes': actual_chunk_size,
                'episode_range': (start_ep, end_ep - 1),
                'size_mb': chunk_size_mb
            })

            # Delete local chunk to free space
            if delete_local_after_upload:
                os.remove(local_chunk_path)
                if verbose:
                    print(f"   ✓ Deleted local chunk (freed {chunk_size_mb:.2f} MB)")

            if verbose:
                print()

        except Exception as e:
            if verbose:
                print(f"   ✗ Failed to create chunk {chunk_idx}: {e}")
            # Clean up partial chunk
            if os.path.exists(local_chunk_path):
                os.remove(local_chunk_path)
            raise

    # Clean up temp directory
    if delete_local_after_upload:
        try:
            shutil.rmtree(local_temp_dir)
            if verbose:
                print(f"✓ Cleaned up temp directory: {local_temp_dir}")
                print()
        except:
            pass

    # Final summary
    if verbose:
        print("=" * 80)
        print("✅ CHUNKED CONSOLIDATION COMPLETE")
        print("=" * 80)
        print(f"Total episodes processed: {successful_episodes}/{num_episodes}")
        print(f"Chunks created: {len(chunk_info)}")
        print(f"Chunk size: {chunk_size} episodes each")
        print(f"Location: s3://{s3_bucket}/{s3_output_prefix}/")
        print()
        print("Chunk details:")
        for info in chunk_info:
            print(f"  chunk_{info['chunk_idx']:04d}.h5: "
                  f"{info['num_episodes']} episodes "
                  f"(ep {info['episode_range'][0]}-{info['episode_range'][1]}), "
                  f"{info['size_mb']:.2f} MB")
        print("=" * 80)

    return {
        'num_chunks': len(chunk_info),
        'total_episodes': successful_episodes,
        'chunk_size': chunk_size,
        'chunks': chunk_info,
        's3_location': f"s3://{s3_bucket}/{s3_output_prefix}/",
        'skipped': False
    }


def get_chunk_info(s3_bucket: str, s3_chunk_prefix: str) -> dict:
    """
    Get information about consolidated chunks in S3.

    Args:
        s3_bucket (str): S3 bucket name
        s3_chunk_prefix (str): S3 prefix containing chunks

    Returns:
        dict: Information about chunks
    """
    fs = s3fs.S3FileSystem()
    s3_path = f"{s3_bucket}/{s3_chunk_prefix}"

    # List chunk files
    try:
        all_files = fs.ls(s3_path)
        chunk_files = [f for f in all_files if f.endswith('.h5')]
        chunk_files.sort()
    except Exception as e:
        raise ValueError(f"Failed to list chunks: {e}")

    if not chunk_files:
        raise ValueError(f"No chunk files found at s3://{s3_path}")

    # Get info from first chunk
    with fs.open(chunk_files[0], 'rb') as f:
        with h5py.File(f, 'r') as h5f:
            first_chunk_episodes = h5f['images'].shape[0]
            max_steps = h5f['images'].shape[1]
            image_shape = h5f['images'].shape[2:]

    # Get total episodes
    total_episodes = 0
    for chunk_file in chunk_files:
        with fs.open(chunk_file, 'rb') as f:
            with h5py.File(f, 'r') as h5f:
                total_episodes += h5f['images'].shape[0]

    return {
        's3_location': f"s3://{s3_bucket}/{s3_chunk_prefix}/",
        'num_chunks': len(chunk_files),
        'total_episodes': total_episodes,
        'episodes_per_chunk': first_chunk_episodes,
        'max_steps': max_steps,
        'image_shape': image_shape,
        'chunk_files': chunk_files
    }
