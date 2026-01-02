"""
Validate the VLM analysis database to ensure data integrity.
Runs all validation checks and reports pass/fail.
"""

import duckdb
import sys


def validate_database(db_path: str = "vlm_analysis.db") -> bool:
    """
    Run all validation checks on the database.

    Args:
        db_path: Path to database file

    Returns:
        True if all validations pass, False otherwise
    """
    with duckdb.connect(db_path, read_only=True) as conn:
        all_passed = True

        print("Running validation checks...\n")

        # Check 1: Row count check
        print("1. Row count check:")
        result = conn.execute("""
            SELECT image_id, COUNT(*) as num_tokens
            FROM image_token_distributions
            GROUP BY image_id
            ORDER BY image_id;
        """).fetchall()

        check_1_passed = True
        for image_id, count in result:
            print(f"   {image_id}: {count:,} tokens")
            # Sanity bounds: At least 256 tokens (1 per position), no more than 100M
            # High counts are expected due to flat probability distributions at some positions
            if count < 256 or count > 100_000_000:
                print(f"   ⚠️  WARNING: Unexpected token count")
                all_passed = False
                check_1_passed = False

        if check_1_passed:
            print("   ✓ Row counts within expected range\n")
        else:
            print()

        # Check 2: Probability sum check
        print("2. Probability sum check:")
        invalid = conn.execute("""
            SELECT image_id, token_position, SUM(probability) as prob_sum
            FROM image_token_distributions
            GROUP BY image_id, token_position
            HAVING prob_sum < 0.9 OR prob_sum > 1.0;
        """).fetchall()

        if invalid:
            print(
                f"   ✗ FAILED: {len(invalid)} positions have invalid probability sums"
            )
            for row in invalid[:5]:  # Show first 5
                print(f"     {row}")
            all_passed = False
        else:
            print("   ✓ All positions have valid probability sums (≥0.9, ≤1.0)\n")

        # Check 3: Ranking consistency check
        print("3. Ranking consistency check:")
        # Check that probability_rank matches actual probability ordering
        # Use token_id as tiebreaker for deterministic ordering
        invalid_ranks = conn.execute("""
            WITH ranked AS (
                SELECT
                    image_id,
                    token_position,
                    probability,
                    probability_rank,
                    ROW_NUMBER() OVER (
                        PARTITION BY image_id, token_position
                        ORDER BY probability DESC, token_id ASC
                    ) as expected_rank
                FROM image_token_distributions
            )
            SELECT COUNT(*) as mismatches
            FROM ranked
            WHERE probability_rank != expected_rank;
        """).fetchone()[0]

        if invalid_ranks > 0:
            print(f"   ✗ FAILED: {invalid_ranks} rows have incorrect probability_rank")
            all_passed = False
        else:
            print("   ✓ All probability_rank values match probability ordering\n")

        # Check 4: Embedding vectors check
        print("4. Embedding vectors check:")
        embedding_count = conn.execute("""
            SELECT COUNT(*) FROM image_token_embeddings;
        """).fetchone()[0]

        expected_embeddings = 6 * 256  # 6 images × 256 positions
        if embedding_count != expected_embeddings:
            print(
                f"   ✗ FAILED: Expected {expected_embeddings} embeddings, found {embedding_count}"
            )
            all_passed = False
        else:
            print(f"   ✓ All {embedding_count} embeddings stored\n")

        # Check 5: Data completeness - all images × positions present
        print("5. Data completeness check:")
        missing = conn.execute("""
            WITH expected AS (
                SELECT DISTINCT image_id FROM image_token_distributions
            )
            SELECT
                image_id,
                COUNT(DISTINCT token_position) as positions
            FROM image_token_distributions
            GROUP BY image_id
            HAVING COUNT(DISTINCT token_position) != 256;
        """).fetchall()

        if missing:
            print(f"   ✗ FAILED: Some images missing positions:")
            for row in missing:
                print(f"     {row}")
            all_passed = False
        else:
            print("   ✓ All images have all 256 positions\n")

        if all_passed:
            print("=" * 50)
            print("✅ ALL VALIDATION CHECKS PASSED")
            print("=" * 50)
            return True
        else:
            print("=" * 50)
            print("❌ SOME VALIDATION CHECKS FAILED")
            print("=" * 50)
            return False


def main():
    db_path = "vlm_analysis.db"
    passed = validate_database(db_path)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
