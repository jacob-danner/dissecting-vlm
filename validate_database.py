"""
Validate the VLM analysis database to ensure data integrity.
Runs all validation checks and reports pass/fail.
"""

import argparse
import duckdb
import sys
from database import parse_filter_method


def validate_distribution_table(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    filter_method: str,
    threshold: float,
) -> bool:
    """
    Validate a single distribution table.

    Args:
        conn: Database connection
        table_name: Name of distribution table to validate
        filter_method: "topp" or "minp"
        threshold: Threshold value

    Returns:
        True if all validations pass, False otherwise
    """
    all_passed = True

    print(f"\n{'=' * 50}")
    print(f"Validating table: {table_name}")
    print(f"{'=' * 50}\n")

    # Check 1: Row count check
    print("1. Row count check:")
    result = conn.execute(f"""
        SELECT image_id, COUNT(*) as num_tokens
        FROM {table_name}
        GROUP BY image_id
        ORDER BY image_id;
    """).fetchall()

    check_1_passed = True
    for image_id, count in result:
        print(f"   {image_id}: {count:,} tokens")
        # Sanity bounds vary by method:
        # - top-p: At least 256 tokens (1 per position guaranteed), no upper bound check
        # - min-p: Could be 0 tokens (all probs < threshold), max 256 * (100 / threshold_percent)
        if filter_method == "topp":
            # Only check lower bound - upper bound depends on probability distribution
            if count < 256:
                print(f"   ⚠️  WARNING: Token count below minimum (expected ≥256)")
                all_passed = False
                check_1_passed = False
        else:  # minp
            min_tokens = 0
            threshold_percent = int(threshold * 100)
            max_tokens = 256 * (100 // threshold_percent)
            if count < min_tokens or count > max_tokens:
                print(
                    f"   ⚠️  WARNING: Token count outside expected range [{min_tokens:,}, {max_tokens:,}]"
                )
                all_passed = False
                check_1_passed = False

    if check_1_passed:
        print("   ✓ Row counts within expected range\n")
    else:
        print()

    # Check 2: Method-specific probability check
    if filter_method == "topp":
        print(f"2. Probability sum check (top-p, threshold={threshold}):")
        invalid = conn.execute(f"""
            SELECT image_id, token_position, SUM(probability) as prob_sum
            FROM {table_name}
            GROUP BY image_id, token_position
            HAVING prob_sum < {threshold} OR prob_sum > 1.0;
        """).fetchall()

        if invalid:
            print(
                f"   ✗ FAILED: {len(invalid)} positions have invalid probability sums"
            )
            for row in invalid[:5]:  # Show first 5
                print(f"     {row}")
            all_passed = False
        else:
            print(
                f"   ✓ All positions have valid probability sums (≥{threshold}, ≤1.0)\n"
            )
    elif filter_method == "minp":
        print(f"2. Individual probability check (min-p, threshold={threshold}):")
        invalid = conn.execute(f"""
            SELECT image_id, token_position, token_id, probability
            FROM {table_name}
            WHERE probability < {threshold} OR probability > 1.0;
        """).fetchall()

        if invalid:
            print(f"   ✗ FAILED: {len(invalid)} tokens have invalid probabilities")
            for row in invalid[:5]:  # Show first 5
                print(f"     {row}")
            all_passed = False
        else:
            print(f"   ✓ All tokens have valid probabilities (≥{threshold}, ≤1.0)\n")

    # Check 3: Ranking consistency check
    print("3. Ranking consistency check:")
    invalid_ranks = conn.execute(f"""
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
            FROM {table_name}
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

    # Check 4: Data completeness - all images × positions present
    print("4. Data completeness check:")
    if filter_method == "topp":
        # For top-p, every position must have at least 1 token (we enforce this in code)
        missing = conn.execute(f"""
            SELECT
                image_id,
                COUNT(DISTINCT token_position) as positions
            FROM {table_name}
            GROUP BY image_id
            HAVING COUNT(DISTINCT token_position) != 256;
        """).fetchall()

        if missing:
            print("   ✗ FAILED: Some images missing positions:")
            for row in missing:
                print(f"     {row}")
            all_passed = False
        else:
            print("   ✓ All images have all 256 positions\n")
    elif filter_method == "minp":
        # For min-p, some positions may have 0 tokens (all probs < threshold)
        # Just verify we have reasonable position coverage
        result = conn.execute(f"""
            SELECT
                image_id,
                COUNT(DISTINCT token_position) as positions
            FROM {table_name}
            GROUP BY image_id
            ORDER BY image_id;
        """).fetchall()

        check_passed = True
        for image_id, positions in result:
            print(f"   {image_id}: {positions} positions with tokens")
            # Sanity check: at least some positions should have tokens
            if positions < 10:
                print(f"     ⚠️  WARNING: Very few positions have tokens")
                all_passed = False
                check_passed = False

        if check_passed:
            print("   ✓ Reasonable position coverage\n")
        else:
            print()

    return all_passed


def validate_embeddings_table(conn: duckdb.DuckDBPyConnection) -> bool:
    """
    Validate the embeddings table.

    Args:
        conn: Database connection

    Returns:
        True if validation passes, False otherwise
    """
    print(f"\n{'=' * 50}")
    print("Validating table: image_token_embeddings")
    print(f"{'=' * 50}\n")

    print("Embedding vectors check:")
    embedding_count = conn.execute("""
        SELECT COUNT(*) FROM image_token_embeddings;
    """).fetchone()[0]

    expected_embeddings = 6 * 256  # 6 images × 256 positions
    if embedding_count != expected_embeddings:
        print(
            f"   ✗ FAILED: Expected {expected_embeddings} embeddings, found {embedding_count}"
        )
        return False
    else:
        print(f"   ✓ All {embedding_count} embeddings stored\n")
        return True


def main():
    parser = argparse.ArgumentParser(description="Validate VLM analysis database")
    parser.add_argument(
        "--filter-method",
        nargs="+",
        required=True,
        help='Filter methods to validate (e.g., "topp_90 minp_1")',
    )
    args = parser.parse_args()

    db_path = "vlm_analysis.db"

    with duckdb.connect(db_path, read_only=True) as conn:
        all_passed = True

        # Validate each distribution table
        for method_str in args.filter_method:
            filter_method, threshold, table_name = parse_filter_method(method_str)

            table_passed = validate_distribution_table(
                conn, table_name, filter_method, threshold
            )
            all_passed = all_passed and table_passed

        # Validate embeddings table once
        embeddings_passed = validate_embeddings_table(conn)
        all_passed = all_passed and embeddings_passed

        # Print final summary
        print(f"\n{'=' * 50}")
        if all_passed:
            print("✅ ALL VALIDATION CHECKS PASSED")
        else:
            print("❌ SOME VALIDATION CHECKS FAILED")
        print(f"{'=' * 50}\n")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
