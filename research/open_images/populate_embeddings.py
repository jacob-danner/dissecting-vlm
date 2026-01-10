"""
Populate Open Images embeddings database.

Extracts vision tower embeddings for Open Images samples and stores them in DuckDB.
Supports batch processing and resumability.

Usage:
    uv run python research/open_images/populate_embeddings.py
    uv run python research/open_images/populate_embeddings.py --batch-size 50
    uv run python research/open_images/populate_embeddings.py --limit 10
"""

import argparse
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

# Add research directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from vlm_tools import load_model_and_processor


DB_PATH = Path(__file__).parent / "open_images_embeddings.db"
IMAGES_DIR = Path(__file__).parent / "sample" / "images"


def create_database(db_path: Path) -> None:
    """Create DuckDB database with embeddings schema if it doesn't exist."""
    with duckdb.connect(str(db_path)) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS image_token_embeddings (
                image_id VARCHAR NOT NULL,
                token_position INTEGER NOT NULL,
                embedding_vector FLOAT[2560] NOT NULL,
                PRIMARY KEY (image_id, token_position)
            );
        """)


def get_processed_image_ids(db_path: Path) -> set[str]:
    """Get set of image IDs already in database."""
    if not db_path.exists():
        return set()

    with duckdb.connect(str(db_path)) as conn:
        result = conn.execute("""
            SELECT DISTINCT image_id FROM image_token_embeddings
        """).fetchall()
        return {row[0] for row in result}


def discover_images(images_dir: Path) -> list[tuple[str, Path]]:
    """Discover all images and return (image_id, path) tuples."""
    images = []
    for path in sorted(images_dir.glob("*.jpg")):
        image_id = path.stem  # filename without extension
        images.append((image_id, path))
    return images


def extract_image_tokens_batch(
    image_paths: list[Path],
    model,
    processor,
) -> torch.Tensor:
    """Extract image tokens for a batch of images.

    Returns:
        Tensor of shape (batch_size, 256, 2560)
    """
    # Load images
    images = [Image.open(path) for path in image_paths]

    # Create messages for each image
    messages_batch = [
        [{"role": "user", "content": [{"type": "image", "image": img}]}]
        for img in images
    ]

    # Process batch through chat template
    inputs = processor.apply_chat_template(
        messages_batch,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to("mps")

    with torch.no_grad():
        # pixel_values shape: (batch_size, channels, height, width)
        image_encoding = model.vision_tower(inputs.pixel_values)
        # image_encoding[0] shape: (batch_size, 256, vision_dim)
        image_tokens = model.multi_modal_projector(image_encoding[0])
        # image_tokens shape: (batch_size, 256, 2560)

    return image_tokens


def insert_embeddings_batch(
    conn: duckdb.DuckDBPyConnection,
    image_ids: list[str],
    embeddings: torch.Tensor,
) -> None:
    """Insert embeddings for a batch of images."""
    rows = []
    # Convert bfloat16 to float32 before numpy conversion
    embeddings_np = embeddings.float().cpu().numpy()

    for batch_idx, image_id in enumerate(image_ids):
        for position in range(256):
            rows.append({
                "image_id": image_id,
                "token_position": position,
                "embedding_vector": embeddings_np[batch_idx, position].tolist(),
            })

    df = pd.DataFrame(rows)
    conn.execute("INSERT INTO image_token_embeddings SELECT * FROM df")


def main():
    parser = argparse.ArgumentParser(
        description="Populate Open Images embeddings database"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of images to process per batch (default: 50)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of images to process (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually processing",
    )
    args = parser.parse_args()

    start_time = time.time()

    # Create database if needed
    print(f"Database: {DB_PATH}")
    create_database(DB_PATH)

    # Get already processed images
    processed_ids = get_processed_image_ids(DB_PATH)
    print(f"Already processed: {len(processed_ids)} images")

    # Discover all images
    all_images = discover_images(IMAGES_DIR)
    print(f"Total images found: {len(all_images)}")

    # Filter out already processed
    images_to_process = [
        (img_id, path) for img_id, path in all_images
        if img_id not in processed_ids
    ]

    # Apply limit if specified
    if args.limit is not None:
        images_to_process = images_to_process[:args.limit]

    print(f"Images to process: {len(images_to_process)}")

    if args.dry_run:
        print("\n[DRY RUN] Would process these images:")
        for img_id, path in images_to_process[:10]:
            print(f"  - {img_id}")
        if len(images_to_process) > 10:
            print(f"  ... and {len(images_to_process) - 10} more")
        return

    if not images_to_process:
        print("Nothing to process!")
        return

    # Load model once
    print("\nLoading model...")
    model, processor = load_model_and_processor()
    print("Model loaded.")

    # Process in batches
    total_processed = 0
    total_skipped = len(processed_ids)

    with duckdb.connect(str(DB_PATH)) as conn:
        # Create batches
        batches = [
            images_to_process[i:i + args.batch_size]
            for i in range(0, len(images_to_process), args.batch_size)
        ]

        pbar = tqdm(batches, desc="Processing batches", unit="batch")
        for batch in pbar:
            image_ids = [img_id for img_id, _ in batch]
            image_paths = [path for _, path in batch]

            # Extract embeddings for batch
            embeddings = extract_image_tokens_batch(image_paths, model, processor)

            # Insert into database
            insert_embeddings_batch(conn, image_ids, embeddings)

            # Commit after each batch (enables resume on crash)
            conn.commit()

            total_processed += len(batch)
            pbar.set_postfix(processed=total_processed)

    elapsed = time.time() - start_time

    print(f"\n{'='*50}")
    print("Summary:")
    print(f"  Total processed: {total_processed}")
    print(f"  Skipped (already in DB): {total_skipped}")
    print(f"  Time elapsed: {elapsed:.1f}s")
    print(f"  Rate: {total_processed / elapsed:.1f} images/sec")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
