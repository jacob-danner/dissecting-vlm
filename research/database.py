import duckdb
import torch
import pandas as pd
from vlm_tools import PositionDistribution


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
        raise ValueError(
            f"Filter method must be 'topp' or 'minp', got: {filter_method}"
        )

    threshold_percent = int(parts[1])
    threshold = threshold_percent / 100.0

    table_name = f"image_token_distributions_{method_str}"

    return filter_method, threshold, table_name


def create_database(
    filter_methods: list[str], db_path: str = "vlm_analysis.db"
) -> None:
    """
    Create DuckDB database with required schema.

    Args:
        filter_methods: List of filter method names (e.g., ["topp_90", "minp_1"])
        db_path: Path to database file
    """
    with duckdb.connect(db_path) as conn:
        # Drop all existing tables (fresh start every time)
        all_tables = conn.execute("SHOW TABLES;").fetchall()
        for (table_name,) in all_tables:
            conn.execute(f"DROP TABLE IF EXISTS {table_name};")

        # Create distribution tables for each filter method
        for method in filter_methods:
            table_name = f"image_token_distributions_{method}"
            conn.execute(f"""
                CREATE TABLE {table_name} (
                    image_id VARCHAR NOT NULL,
                    token_position INTEGER NOT NULL,
                    token_id INTEGER NOT NULL,
                    token_text VARCHAR NOT NULL,
                    probability FLOAT NOT NULL,
                    probability_rank INTEGER NOT NULL,
                    PRIMARY KEY (image_id, token_position, token_id)
                );
            """)

        # Create image_token_embeddings table (shared across all filter methods)
        conn.execute("""
            CREATE TABLE image_token_embeddings (
                image_id VARCHAR NOT NULL,
                token_position INTEGER NOT NULL,
                embedding_vector FLOAT[2560] NOT NULL,
                PRIMARY KEY (image_id, token_position)
            );
        """)


def insert_image_distributions(
    db_path: str,
    table_name: str,
    image_id: str,
    distributions: list[PositionDistribution],
    embeddings: torch.Tensor | None = None,
) -> None:
    """
    Insert image token distributions and optionally embeddings into database.

    Args:
        db_path: Path to database file
        table_name: Name of distribution table to insert into
        image_id: Unique identifier for image
        distributions: List of PositionDistribution objects from extract_image_token_distributions
        embeddings: Optional image token embedding vectors (256, d_model). Only insert once per image.
    """
    with duckdb.connect(db_path) as conn:
        # Prepare distribution data
        distribution_rows = []
        for dist in distributions:
            for token in dist.tokens:
                distribution_rows.append(
                    {
                        "image_id": image_id,
                        "token_position": dist.position,
                        "token_id": token.token_id,
                        "token_text": token.token_text,
                        "probability": token.probability,
                        "probability_rank": token.rank,
                    }
                )

        # Insert distributions
        if distribution_rows:
            distribution_df = pd.DataFrame(distribution_rows)  # noqa: F841
            conn.execute(f"""
                INSERT INTO {table_name}
                SELECT * FROM distribution_df;
            """)

        # Insert embeddings only if provided
        if embeddings is not None:
            # Prepare embedding data
            # Convert bfloat16 to float32 before numpy conversion
            embeddings_cpu = embeddings.float().cpu().numpy()
            embedding_rows = []
            for position in range(embeddings.shape[0]):
                embedding_rows.append(
                    {
                        "image_id": image_id,
                        "token_position": position,
                        "embedding_vector": embeddings_cpu[position].tolist(),
                    }
                )

            # Insert embeddings
            if embedding_rows:
                embedding_df = pd.DataFrame(embedding_rows)  # noqa: F841
                conn.execute("""
                    INSERT INTO image_token_embeddings
                    SELECT * FROM embedding_df;
                """)
