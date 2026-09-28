# Data Preparation Module

This module handles data collection and preprocessing for the JEPA CartPole project, with integrated S3 support to avoid local disk space issues on SageMaker.

## Structure

```
data_prep/
├── __init__.py          # Module exports
├── environment.py       # Environment creation and configuration
├── collector.py         # Parallel data collection with S3 support
├── merge.py            # Episode consolidation from S3 or local files
└── visualization.py    # Dataset visualization tools
```

## Key Components

### Environment (`environment.py`)
- `create_cartpole_env()`: Creates CartPole environment with pixel observations
- `setup_pygame_headless()`: Configures headless rendering
- `get_observation_shape()`: Returns observation dimensions

### Collector (`collector.py`)
- `collect_data()`: Main function for parallel episode collection with S3 support
- `collect_episode()`: Collects a single episode
- `collect_data_chunk()`: Worker process for parallel collection

### Merge (`merge.py`)
- `merge_episode_files()`: Consolidates temporary files into single HDF5
- `check_dataset_exists()`: Verifies if final dataset already exists
- `get_dataset_info()`: Retrieves dataset statistics

### Visualization (`visualization.py`)
- `plot_random_samples()`: Display random frames from dataset
- `plot_episode_sequence()`: Show temporal sequence from episode
- `plot_dataset_statistics()`: Statistical analysis plots
- `print_dataset_summary()`: Comprehensive dataset summary

## S3 Storage Strategy

### Why S3?

SageMaker instances have limited local disk space. Collecting 500 episodes of CartPole with RGB images (~25 GB) can fill the disk quickly. Our solution:

1. **Batch Collection**: Collect episodes in small batches (e.g., 50 episodes = ~2.5 GB)
2. **Immediate Upload**: Upload each batch to S3 and delete local files
3. **Final Consolidation**: Download all files from S3, merge, and clean up

### Benefits

- ✅ **Local disk usage**: Only 2-3 GB peak (one batch + final dataset)
- ✅ **S3 temporary storage**: Automatically cleaned after consolidation
- ✅ **Cost effective**: S3 storage costs pennies, SageMaker disk is expensive
- ✅ **Smart caching**: Skip collection if files already exist in S3 or locally

## Usage Examples

### Basic Collection with S3

```python
from jepacartpole.data_prep import collect_data, merge_episode_files

# Collect episodes and save to S3 in batches
collect_data(
    num_episodes=500,
    max_steps=500,
    output_dir='../data/raw',
    use_s3=True,
    s3_bucket='jepa-cartpole-tcc-seu-nome',
    s3_prefix='tmp/raw_data',
    batch_size=50  # Upload to S3 every 50 episodes
)

# Consolidate from S3
merge_episode_files(
    episodes_dir='../data/raw',
    output_file='../data/vae/vae_data.h5',
    max_steps=500,
    use_s3=True,
    s3_bucket='jepa-cartpole-tcc-seu-nome',
    s3_prefix='tmp/raw_data',
    delete_temp_files=True  # Clean up S3 files after merge
)
```

### With Smart Caching

```python
from jepacartpole.data_prep import check_dataset_exists, collect_data

# Check if dataset already exists
if not check_dataset_exists('data/vae/vae_data.h5'):
    # Check if files exist in S3
    from jepacartpole.storage import S3Manager
    s3 = S3Manager('jepa-cartpole-tcc-seu-nome')
    
    if s3.get_file_count('tmp/raw_data') == 0:
        # No dataset, no S3 files - collect from scratch
        collect_data(
            num_episodes=500,
            max_steps=500,
            use_s3=True,
            s3_bucket='jepa-cartpole-tcc-seu-nome',
            s3_prefix='tmp/raw_data'
        )
    
    # Consolidate (from S3 or local)
    merge_episode_files(
        episodes_dir='../data/raw',
        output_file='data/vae/vae_data.h5',
        max_steps=500,
        use_s3=True,
        s3_bucket='jepa-cartpole-tcc-seu-nome',
        s3_prefix='tmp/raw_data'
    )
```

### Local-Only (No S3)

For smaller datasets or when disk space isn't an issue:

```python
collect_data(
    num_episodes=100,
    max_steps=500,
    output_dir='../data/raw',
    use_s3=False  # No S3, store locally
)

merge_episode_files(
    episodes_dir='../data/raw',
    output_file='../data/vae/vae_data.h5',
    max_steps=500,
    use_s3=False
)
```

## Performance Notes

### Parallel Collection with Real-Time Progress

The data collection uses **chunked parallel processing** with **real-time per-worker tracking**:

#### Features
- **All CPUs utilized**: Uses all available CPU cores by default
- **Chunked work distribution**: Work split into small chunks (default: 10 episodes)
- **Real-time progress tracking**: See exact episode counts per worker
- **Live progress bar**: Updates as each episode completes across all workers

#### Progress Display

```
Collecting episodes: |████████| 500/500 [02:30<00:00] W0:252 | W1:248

📤 Uploading batch to S3 (100 episodes collected)...
✅ Uploaded 50 files, freed local disk space

Collecting episodes: |████████| 500/500 [05:00<00:00] W0:252 | W1:248

✅ Data collection completed!
   Total episodes collected: 500
   Episodes by worker:
      Worker 0: 252 episodes
      Worker 1: 248 episodes
   Files saved to: s3://bucket/tmp/raw_data
```

**Progress bar format:**
- Total episodes completed / Total episodes
- Time elapsed and estimated time remaining
- `W0:X | W1:Y` - Episodes completed by each worker in real-time
- Batch upload notifications as data moves to S3

### Performance on 2-CPU System (500 episodes)

**Without S3** (Local only):
- Collection: ~2.5 minutes
- Disk usage: ~25 GB
- ❌ Can fail if disk is full

**With S3** (Recommended):
- Collection: ~3 minutes (slight overhead from uploads)
- Peak disk usage: ~3 GB (one batch + overhead)
- ✅ Works even with limited disk space
- S3 upload time: ~30 seconds per 50-episode batch
- Total S3 uploads: 10 batches (500 episodes / 50)

## Dataset Structure

The consolidated HDF5 dataset has the following structure:

```
vae_data.h5
├── images:  uint8   (num_episodes, max_steps, 400, 600, 3)
├── actions: float32 (num_episodes, max_steps)
├── rewards: float32 (num_episodes, max_steps)
└── dones:   bool    (num_episodes, max_steps)
```

## Troubleshooting

### Disk Full During Collection

**Solution**: Use S3 storage with smaller batch size:

```python
collect_data(
    ...,
    use_s3=True,
    batch_size=25  # Smaller batches = less local disk usage
)
```

### S3 Access Denied

**Solution**: Ensure SageMaker role has S3 permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::jepa-cartpole-tcc-seu-nome/*",
        "arn:aws:s3:::jepa-cartpole-tcc-seu-nome"
      ]
    }
  ]
}
```

### Files Already Exist in S3

The system automatically detects existing S3 files and skips collection:

```
⚠️  Found 500 existing files in S3
   s3://jepa-cartpole-tcc-seu-nome/tmp/raw_data
   Skipping data collection - files already exist!
```

To force recollection, delete S3 files first:

```python
from jepacartpole.storage import delete_s3_files

delete_s3_files('jepa-cartpole-tcc-seu-nome', 'tmp/raw_data')
```

## Configuration Tips

### Optimal Batch Size

- **2-4 GB free disk**: `batch_size=25` (conservative)
- **5-10 GB free disk**: `batch_size=50` (balanced)
- **>10 GB free disk**: `batch_size=100` (faster, fewer uploads)

### Episode Count vs Disk Space

| Episodes | Image Size | Total Size | Local Peak (batch=50) |
|----------|-----------|------------|-----------------------|
| 100      | 720 KB    | ~5 GB      | ~1 GB                |
| 500      | 720 KB    | ~25 GB     | ~3 GB                |
| 1000     | 720 KB    | ~50 GB     | ~5 GB                |

## Cost Estimation

### S3 Storage Costs (us-east-1)

- **Storage**: $0.023 per GB-month
- **PUT requests**: $0.005 per 1,000 requests
- **GET requests**: $0.0004 per 1,000 requests
- **DELETE**: Free

**Example** (500 episodes, 25 GB, 1 hour collection):
- Storage: 25 GB × $0.023 / 730 hours = $0.0008 per hour ≈ **$0.001**
- PUT: 500 files × $0.005 / 1000 = **$0.0025**
- GET: 500 files × $0.0004 / 1000 = **$0.0002**
- **Total**: ~**$0.004** (less than 1 cent!)

### Comparison

- **SageMaker disk**: $0.10 per GB-month (additional disk)
- **S3 temporary storage**: $0.004 for entire collection
- **Savings**: ~99% cheaper for temporary storage

## See Also

- `jepacartpole/storage.py` - S3 utilities for file operations
- `notebooks/1-data_prep/1-data_collection.ipynb` - Complete workflow example
