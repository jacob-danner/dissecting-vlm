from dataclasses import dataclass
import duckdb
import torch
import pandas as pd
from vlm_tools import PositionDistribution


@dataclass
class PositionTokenStats:
    """Statistics for a token at a specific position across images."""

    token_text: str
    avg_probability: float
    frequency: int


def create_database(db_path: str = "vlm_analysis.db") -> None:
    """
    Create DuckDB database with required schema.

    Args:
        db_path: Path to database file
    """
    with duckdb.connect(db_path) as conn:
        # Drop tables if they exist (overwrite on each run)
        conn.execute("DROP TABLE IF EXISTS image_token_distributions;")
        conn.execute("DROP TABLE IF EXISTS image_token_embeddings;")

        # Create image_token_distributions table
        conn.execute("""
            CREATE TABLE image_token_distributions (
                image_id VARCHAR NOT NULL,
                token_position INTEGER NOT NULL,
                token_id INTEGER NOT NULL,
                token_text VARCHAR NOT NULL,
                probability FLOAT NOT NULL,
                probability_rank INTEGER NOT NULL,
                PRIMARY KEY (image_id, token_position, token_id)
            );
        """)

        # Create image_token_embeddings table
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
    image_id: str,
    distributions: list[PositionDistribution],
    embeddings: torch.Tensor,
) -> None:
    """
    Insert image token distributions and embeddings into database.

    Args:
        db_path: Path to database file
        image_id: Unique identifier for image
        distributions: List of PositionDistribution objects from extract_image_token_distributions
        embeddings: Image token embedding vectors (256, d_model)
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
            distribution_df = pd.DataFrame(distribution_rows)
            conn.execute("""
                INSERT INTO image_token_distributions
                SELECT * FROM distribution_df;
            """)

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
            embedding_df = pd.DataFrame(embedding_rows)
            conn.execute("""
                INSERT INTO image_token_embeddings
                SELECT * FROM embedding_df;
            """)


def query_position_tokens(
    db_path: str, position: int, top_k: int = 10
) -> list[PositionTokenStats]:
    """
    Query most common tokens at a given position across all images.

    Args:
        db_path: Path to database file
        position: Token position (0-255)
        top_k: Number of top tokens to return

    Returns:
        List of PositionTokenStats objects
    """
    with duckdb.connect(db_path, read_only=True) as conn:
        result = conn.execute(
            """
            SELECT
                token_text,
                AVG(probability) as avg_probability,
                COUNT(*) as frequency
            FROM image_token_distributions
            WHERE token_position = ?
            GROUP BY token_text
            ORDER BY frequency DESC, avg_probability DESC
            LIMIT ?;
        """,
            [position, top_k],
        ).fetchall()

        return [
            PositionTokenStats(
                token_text=row[0], avg_probability=row[1], frequency=row[2]
            )
            for row in result
        ]
