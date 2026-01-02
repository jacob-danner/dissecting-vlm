"""
Populate the VLM analysis database with all repo images.
"""

import argparse
from vlm_tools import load_model_and_processor, extract_image_token_distributions
from database import create_database, insert_image_distributions
from pathlib import Path


def parse_filter_method(method_str: str) -> tuple[str, float, str]:
    """
    Parse filter method string into components.

    Args:
        method_str: Format "method_thresholdpercent" (e.g., "topp_90", "minp_1")

    Returns:
        filter_method: "topp" or "minp"
        threshold: Threshold value (e.g., 0.90, 0.01)
        table_name: Full table name (e.g., "image_token_distributions_topp_90")
    """
    parts = method_str.split("_")
    if len(parts) != 2:
        raise ValueError(f"Invalid filter method format: {method_str}")

    filter_method = parts[0]
    if filter_method not in ["topp", "minp"]:
        raise ValueError(f"Filter method must be 'topp' or 'minp', got: {filter_method}")

    threshold_percent = int(parts[1])
    threshold = threshold_percent / 100.0

    table_name = f"image_token_distributions_{method_str}"

    return filter_method, threshold, table_name


def main():
    parser = argparse.ArgumentParser(
        description="Populate VLM analysis database with filtered probability distributions"
    )
    parser.add_argument(
        "--filter-method",
        nargs="+",
        required=True,
        help='Filter methods to populate (e.g., "topp_90 minp_1")',
    )
    args = parser.parse_args()

    # Image paths
    images = [
        "images/monocolor.png",
        "images/bicolor.png",
        "images/sunrise_mountain.jpg",
        "images/chicago_l.jpg",
        "images/code_snippet.png",
        "images/vision_language_model_data_flow.png",
    ]

    # Initialize database
    db_path = "vlm_analysis.db"
    create_database(args.filter_method, db_path)
    print(f"Created database with tables: {args.filter_method}\n")

    # Load model once
    model, processor = load_model_and_processor()

    # Process each image
    for image_path in images:
        print(f"Processing {image_path}...")
        image_id = Path(image_path).name

        # Extract embeddings once per image
        embeddings = None
        for idx, method_str in enumerate(args.filter_method):
            filter_method, threshold, table_name = parse_filter_method(method_str)

            # Extract distributions with current filter method
            emb, distributions = extract_image_token_distributions(
                image_path, model, processor, filter_method, threshold
            )

            # Store embeddings from first method
            if embeddings is None:
                embeddings = emb

            # Insert distributions into method-specific table
            # Only insert embeddings on first method to avoid duplicates
            insert_image_distributions(
                db_path, table_name, image_id, distributions,
                embeddings if idx == 0 else None
            )
            total_tokens = sum(len(d.tokens) for d in distributions)
            print(f"  {method_str}: Inserted {total_tokens} total tokens")

        print()

    print(f"Database populated: {db_path}")


if __name__ == "__main__":
    main()
