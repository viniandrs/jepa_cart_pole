"""
Storage utilities for S3 integration.

This module provides functions for uploading and downloading files to/from S3,
used throughout the project to manage large datasets without filling local disk.
"""

import os
import boto3
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm


class S3Manager:
    """
    Manager for S3 operations with progress tracking.

    Handles uploading, downloading, listing, and deleting files from S3.
    """

    def __init__(self, bucket_name: str, verbose: bool = True):
        """
        Initialize S3 manager.

        Args:
            bucket_name (str): Name of the S3 bucket
            verbose (bool): If True, print progress messages
        """
        self.bucket_name = bucket_name
        self.verbose = verbose
        self.s3_client = boto3.client('s3')
        self.s3_resource = boto3.resource('s3')
        self.bucket = self.s3_resource.Bucket(bucket_name)

    def upload_file(self, local_path: str, s3_key: str) -> bool:
        """
        Upload a single file to S3.

        Args:
            local_path (str): Path to local file
            s3_key (str): S3 key (path in bucket)

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Remove leading slash from s3_key if present
            s3_key = s3_key.lstrip('/')

            file_size = os.path.getsize(local_path)

            if self.verbose:
                print(f"Uploading {Path(local_path).name} ({file_size / 1024 / 1024:.2f} MB) to S3...")

            self.s3_client.upload_file(local_path, self.bucket_name, s3_key)

            if self.verbose:
                print(f"✓ Uploaded to s3://{self.bucket_name}/{s3_key}")

            return True

        except Exception as e:
            if self.verbose:
                print(f"✗ Error uploading {local_path}: {e}")
            return False

    def upload_files(self, local_paths: List[str], s3_prefix: str,
                    delete_after_upload: bool = True) -> int:
        """
        Upload multiple files to S3 with progress tracking.

        Args:
            local_paths (List[str]): List of local file paths
            s3_prefix (str): S3 prefix (directory) for all files
            delete_after_upload (bool): If True, delete local files after successful upload

        Returns:
            int: Number of files successfully uploaded
        """
        # Remove leading/trailing slashes from prefix
        s3_prefix = s3_prefix.strip('/')

        if self.verbose:
            print(f"\n📤 Uploading {len(local_paths)} files to S3")
            print(f"   Bucket: {self.bucket_name}")
            print(f"   Prefix: {s3_prefix}\n")

        uploaded_count = 0
        iterator = tqdm(local_paths, desc="Uploading files", unit="file") if self.verbose else local_paths

        for local_path in iterator:
            filename = Path(local_path).name
            s3_key = f"{s3_prefix}/{filename}"

            if self.upload_file(local_path, s3_key):
                uploaded_count += 1

                # Delete local file after successful upload
                if delete_after_upload:
                    try:
                        os.remove(local_path)
                        if self.verbose and not isinstance(iterator, tqdm):
                            print(f"  Deleted local file: {filename}")
                    except Exception as e:
                        if self.verbose:
                            print(f"  Warning: Could not delete {filename}: {e}")

        if self.verbose:
            print(f"\n✅ Uploaded {uploaded_count}/{len(local_paths)} files to S3")
            if delete_after_upload:
                print(f"   Local files deleted to save disk space")

        return uploaded_count

    def download_file(self, s3_key: str, local_path: str) -> bool:
        """
        Download a single file from S3.

        Args:
            s3_key (str): S3 key (path in bucket)
            local_path (str): Local path to save file

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Remove leading slash from s3_key if present
            s3_key = s3_key.lstrip('/')

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            if self.verbose:
                print(f"Downloading {Path(s3_key).name} from S3...")

            self.s3_client.download_file(self.bucket_name, s3_key, local_path)

            if self.verbose:
                file_size = os.path.getsize(local_path)
                print(f"✓ Downloaded to {local_path} ({file_size / 1024 / 1024:.2f} MB)")

            return True

        except Exception as e:
            if self.verbose:
                print(f"✗ Error downloading {s3_key}: {e}")
            return False

    def download_files(self, s3_prefix: str, local_dir: str) -> int:
        """
        Download all files with a given prefix from S3.

        Args:
            s3_prefix (str): S3 prefix (directory) to download from
            local_dir (str): Local directory to save files

        Returns:
            int: Number of files successfully downloaded
        """
        # Remove leading/trailing slashes from prefix
        s3_prefix = s3_prefix.strip('/')

        # List all objects with this prefix
        s3_files = self.list_files(s3_prefix)

        if not s3_files:
            if self.verbose:
                print(f"No files found with prefix: {s3_prefix}")
            return 0

        if self.verbose:
            print(f"\n📥 Downloading {len(s3_files)} files from S3")
            print(f"   Bucket: {self.bucket_name}")
            print(f"   Prefix: {s3_prefix}")
            print(f"   Destination: {local_dir}\n")

        os.makedirs(local_dir, exist_ok=True)

        downloaded_count = 0
        iterator = tqdm(s3_files, desc="Downloading files", unit="file") if self.verbose else s3_files

        for s3_key in iterator:
            filename = Path(s3_key).name
            local_path = os.path.join(local_dir, filename)

            if self.download_file(s3_key, local_path):
                downloaded_count += 1

        if self.verbose:
            print(f"\n✅ Downloaded {downloaded_count}/{len(s3_files)} files from S3")

        return downloaded_count

    def list_files(self, s3_prefix: str) -> List[str]:
        """
        List all files with a given prefix in S3.

        Args:
            s3_prefix (str): S3 prefix (directory) to list

        Returns:
            List[str]: List of S3 keys
        """
        # Remove leading/trailing slashes from prefix
        s3_prefix = s3_prefix.strip('/')

        try:
            objects = self.bucket.objects.filter(Prefix=s3_prefix)
            s3_files = [obj.key for obj in objects if obj.key.endswith('.h5') or obj.key.endswith('.hdf5')]
            return s3_files

        except Exception as e:
            if self.verbose:
                print(f"Error listing files from S3: {e}")
            return []

    def file_exists(self, s3_key: str) -> bool:
        """
        Check if a file exists in S3.

        Args:
            s3_key (str): S3 key to check

        Returns:
            bool: True if file exists, False otherwise
        """
        # Remove leading slash from s3_key if present
        s3_key = s3_key.lstrip('/')

        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except:
            return False

    def delete_file(self, s3_key: str) -> bool:
        """
        Delete a single file from S3.

        Args:
            s3_key (str): S3 key to delete

        Returns:
            bool: True if successful, False otherwise
        """
        # Remove leading slash from s3_key if present
        s3_key = s3_key.lstrip('/')

        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            if self.verbose:
                print(f"✓ Deleted s3://{self.bucket_name}/{s3_key}")
            return True

        except Exception as e:
            if self.verbose:
                print(f"✗ Error deleting {s3_key}: {e}")
            return False

    def delete_files(self, s3_prefix: str) -> int:
        """
        Delete all files with a given prefix from S3.

        Args:
            s3_prefix (str): S3 prefix (directory) to delete from

        Returns:
            int: Number of files successfully deleted
        """
        # Remove leading/trailing slashes from prefix
        s3_prefix = s3_prefix.strip('/')

        s3_files = self.list_files(s3_prefix)

        if not s3_files:
            if self.verbose:
                print(f"No files to delete with prefix: {s3_prefix}")
            return 0

        if self.verbose:
            print(f"\n🗑️  Deleting {len(s3_files)} files from S3")
            print(f"   Bucket: {self.bucket_name}")
            print(f"   Prefix: {s3_prefix}\n")

        deleted_count = 0
        iterator = tqdm(s3_files, desc="Deleting files", unit="file") if self.verbose else s3_files

        for s3_key in iterator:
            if self.delete_file(s3_key):
                deleted_count += 1

        if self.verbose:
            print(f"\n✅ Deleted {deleted_count}/{len(s3_files)} files from S3")

        return deleted_count

    def get_file_count(self, s3_prefix: str) -> int:
        """
        Count files with a given prefix in S3.

        Args:
            s3_prefix (str): S3 prefix to count

        Returns:
            int: Number of files
        """
        return len(self.list_files(s3_prefix))


# Convenience functions for backward compatibility
def upload_to_s3(local_paths: List[str], bucket: str, s3_prefix: str,
                delete_after_upload: bool = True, verbose: bool = True) -> int:
    """
    Upload files to S3 (convenience function).

    Args:
        local_paths (List[str]): List of local file paths
        bucket (str): S3 bucket name
        s3_prefix (str): S3 prefix (directory)
        delete_after_upload (bool): Delete local files after upload
        verbose (bool): Print progress messages

    Returns:
        int: Number of files uploaded
    """
    manager = S3Manager(bucket, verbose=verbose)
    return manager.upload_files(local_paths, s3_prefix, delete_after_upload)


def download_from_s3(bucket: str, s3_prefix: str, local_dir: str,
                    verbose: bool = True) -> int:
    """
    Download files from S3 (convenience function).

    Args:
        bucket (str): S3 bucket name
        s3_prefix (str): S3 prefix (directory)
        local_dir (str): Local directory to save files
        verbose (bool): Print progress messages

    Returns:
        int: Number of files downloaded
    """
    manager = S3Manager(bucket, verbose=verbose)
    return manager.download_files(s3_prefix, local_dir)


def list_s3_files(bucket: str, s3_prefix: str, verbose: bool = False) -> List[str]:
    """
    List files in S3 (convenience function).

    Args:
        bucket (str): S3 bucket name
        s3_prefix (str): S3 prefix (directory)
        verbose (bool): Print progress messages

    Returns:
        List[str]: List of S3 keys
    """
    manager = S3Manager(bucket, verbose=verbose)
    return manager.list_files(s3_prefix)


def delete_s3_files(bucket: str, s3_prefix: str, verbose: bool = True) -> int:
    """
    Delete files from S3 (convenience function).

    Args:
        bucket (str): S3 bucket name
        s3_prefix (str): S3 prefix (directory)
        verbose (bool): Print progress messages

    Returns:
        int: Number of files deleted
    """
    manager = S3Manager(bucket, verbose=verbose)
    return manager.delete_files(s3_prefix)
