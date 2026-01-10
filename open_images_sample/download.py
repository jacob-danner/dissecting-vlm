"""Download a random sample of images from the Open Images V7 test set.

This script samples n random images from the ~125k test set images and downloads
them from S3. Sampling is reproducible via --seed.

Prerequisites:
    Download 'test-images-with-rotation.csv' from Open Images V7:
    https://storage.googleapis.com/openimages/web/download_v7.html
    Place it in the open_images_sample/ directory (same directory as this script).

Usage:
    python open_images_sample/download.py --n 100 --seed 42
"""

import argparse
import csv
import random
import sys
from concurrent import futures
from pathlib import Path

import boto3
import botocore
import tqdm

BUCKET_NAME = "open-images-dataset"
SCRIPT_DIR = Path(__file__).parent
SOURCE_CSV = SCRIPT_DIR / "test-images-with-rotation.csv"
IMAGES_DIR = SCRIPT_DIR / "images"
METADATA_CSV = SCRIPT_DIR / "metadata.csv"


def load_source_csv() -> list[dict]:
    """Load the source CSV file containing all test images."""
    if not SOURCE_CSV.exists():
        sys.exit(f"ERROR: Source CSV not found at {SOURCE_CSV}")

    with open(SOURCE_CSV, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def sample_images(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Randomly sample n images with reproducible seed."""
    random.seed(seed)
    if n > len(rows):
        print(
            f"WARNING: Requested {n} images but only {len(rows)} available. Using all."
        )
        n = len(rows)
    return random.sample(rows, n)


def get_existing_images() -> set[str]:
    """Get set of already-downloaded image IDs."""
    if not IMAGES_DIR.exists():
        return set()
    return {p.stem for p in IMAGES_DIR.glob("*.jpg")}


def download_one_image(
    bucket, image_id: str, split: str = "test"
) -> tuple[str, bool, str]:
    """Download a single image from S3.

    Returns:
        Tuple of (image_id, success, error_message)
    """
    dest_path = IMAGES_DIR / f"{image_id}.jpg"
    try:
        bucket.download_file(f"{split}/{image_id}.jpg", str(dest_path))
        return (image_id, True, "")
    except botocore.exceptions.ClientError as e:
        return (image_id, False, str(e))


def download_images(
    rows: list[dict], num_workers: int = 5
) -> tuple[list[dict], list[tuple[str, str]]]:
    """Download images, skipping already-downloaded ones.

    Returns:
        Tuple of (successfully_downloaded_rows, failed_downloads)
    """
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    existing = get_existing_images()
    to_download = [(r, r["ImageID"]) for r in rows if r["ImageID"] not in existing]
    already_have = [r for r in rows if r["ImageID"] in existing]

    if already_have:
        print(f"Skipping {len(already_have)} already-downloaded images")

    if not to_download:
        print("All images already downloaded")
        return already_have, []

    print(f"Downloading {len(to_download)} images...")

    bucket = boto3.resource(
        "s3", config=botocore.config.Config(signature_version=botocore.UNSIGNED)
    ).Bucket(BUCKET_NAME)

    successful_rows = list(already_have)
    failed_downloads = []

    progress_bar = tqdm.tqdm(total=len(to_download), desc="Downloading", leave=True)

    with futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_row = {
            executor.submit(download_one_image, bucket, image_id): row
            for row, image_id in to_download
        }

        for future in futures.as_completed(future_to_row):
            row = future_to_row[future]
            image_id, success, error = future.result()

            if success:
                successful_rows.append(row)
            else:
                failed_downloads.append((image_id, error))
                print(f"\nFailed: {image_id} - {error}")

            progress_bar.update(1)

    progress_bar.close()
    return successful_rows, failed_downloads


def write_metadata(rows: list[dict], fieldnames: list[str]) -> None:
    """Write metadata CSV for successfully downloaded images."""
    with open(METADATA_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote metadata for {len(rows)} images to {METADATA_CSV}")


def main():
    parser = argparse.ArgumentParser(
        description="Download random sample of Open Images V7 test images"
    )
    parser.add_argument(
        "--n", type=int, required=True, help="Number of images to sample"
    )
    parser.add_argument(
        "--seed", type=int, required=True, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--workers", type=int, default=5, help="Number of parallel download workers"
    )
    args = parser.parse_args()

    print(f"Loading source CSV from {SOURCE_CSV}...")
    all_rows = load_source_csv()
    fieldnames = list(all_rows[0].keys()) if all_rows else []
    print(f"Found {len(all_rows)} images in source CSV")

    print(f"Sampling {args.n} images with seed {args.seed}...")
    sampled = sample_images(all_rows, args.n, args.seed)

    successful, failed = download_images(sampled, args.workers)

    write_metadata(successful, fieldnames)

    print(f"\nSummary:")
    print(f"  Requested: {args.n}")
    print(f"  Downloaded: {len(successful)}")
    print(f"  Failed: {len(failed)}")

    if failed:
        print("\nFailed downloads:")
        for image_id, error in failed:
            print(f"  {image_id}: {error}")


if __name__ == "__main__":
    main()
