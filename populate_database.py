"""
Populate the VLM analysis database with all repo images.
"""

from vlm_tools import load_model_and_processor, extract_image_token_distributions
from database import create_database, insert_image_distributions
from pathlib import Path


def main():
    # Image paths
    images = [
        "images/monocolor.png",
        "images/bicolor.png",
        "images/sunrise_mountain.jpg",
        "images/chicago_l.jpg",
        "images/code_snippet.png",
        "images/vision_language_model_data_flow.png",
    ]

    # Initialize
    db_path = "vlm_analysis.db"
    create_database(db_path)
    model, processor = load_model_and_processor()

    # Process each image
    for image_path in images:
        print(f"Processing {image_path}...")
        image_id = Path(image_path).name
        embeddings, distributions = extract_image_token_distributions(
            image_path, model, processor, top_p=0.9
        )
        insert_image_distributions(db_path, image_id, distributions, embeddings)
        total_tokens = sum(len(d.tokens) for d in distributions)
        print(f"  Inserted {total_tokens} total tokens")

    print(f"\nDatabase populated: {db_path}")


if __name__ == "__main__":
    main()
