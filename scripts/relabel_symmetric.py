"""
Relabel existing patterns with symmetric success/failure criteria.

This script updates the outcome labels for existing detected patterns
without re-running the full breakout detection pipeline.

Usage:
    python scripts/relabel_symmetric.py
    python scripts/relabel_symmetric.py --threshold 10  # +/-10%
    python scripts/relabel_symmetric.py --dry-run       # Show stats without updating
"""

import argparse
import logging
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from scanner.config import get
from scanner.db import init_db, get_connection, get_cursor
from scanner.data_pipeline import get_price_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def relabel_patterns_symmetric(
    threshold_pct: float = 10.0,
    outcome_window_weeks: int = 8,
    dry_run: bool = False,
) -> dict:
    """Relabel all patterns using symmetric thresholds.

    Args:
        threshold_pct: Symmetric threshold (e.g., 10 for +/-10%)
        outcome_window_weeks: Weeks to track outcome
        dry_run: If True, don't update database, just show stats

    Returns:
        Dict with relabeling statistics
    """
    init_db()
    conn = get_connection()

    outcome_window_days = outcome_window_weeks * 5  # Trading days

    # Get all patterns
    df = pd.read_sql_query(
        """
        SELECT id, symbol, pivot_date, pivot_price, outcome,
               outcome_return_pct, outcome_max_gain_pct, outcome_max_loss_pct
        FROM detected_patterns
        ORDER BY pivot_date
        """,
        conn,
    )

    logger.info(f"Found {len(df)} patterns to relabel with +/-{threshold_pct}% threshold")

    # Count original outcomes
    original_counts = df["outcome"].value_counts().to_dict()
    logger.info(f"Original outcomes: {original_counts}")

    # Process each pattern
    updates = []
    stats = {
        "success": 0,
        "failure": 0,
        "neutral": 0,
        "pending": 0,
        "unchanged": 0,
        "changed": 0,
    }

    # Cache price data per symbol
    price_cache = {}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Relabeling"):
        symbol = row["symbol"]
        pivot_date = row["pivot_date"]
        pivot_price = row["pivot_price"]
        original_outcome = row["outcome"]

        # Get price data (cached)
        if symbol not in price_cache:
            stock_df = get_price_data(symbol)
            if stock_df.empty:
                continue
            stock_df["date_str"] = stock_df["date"].apply(
                lambda d: d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)[:10]
            )
            price_cache[symbol] = stock_df

        stock_df = price_cache[symbol]

        # Find breakout index
        breakout_idx = stock_df[stock_df["date_str"] == pivot_date].index
        if len(breakout_idx) == 0:
            continue

        idx = breakout_idx[0]
        end_idx = min(idx + outcome_window_days, len(stock_df))

        if idx >= len(stock_df) - 1:
            new_outcome = "pending"
        else:
            future_df = stock_df.iloc[idx + 1 : end_idx]
            if future_df.empty:
                new_outcome = "pending"
            else:
                # Check for symmetric thresholds
                hit_success = False
                hit_failure = False

                for _, price_row in future_df.iterrows():
                    day_gain = ((price_row["high"] - pivot_price) / pivot_price) * 100
                    day_loss = ((pivot_price - price_row["low"]) / pivot_price) * 100

                    # Symmetric: same threshold for both
                    if day_gain >= threshold_pct and not hit_failure:
                        hit_success = True
                        break
                    if day_loss >= threshold_pct:
                        hit_failure = True
                        break

                if hit_success:
                    new_outcome = "success"
                elif hit_failure:
                    new_outcome = "failure"
                else:
                    # Use final return to determine
                    final_price = future_df.iloc[-1]["close"]
                    final_return = ((final_price - pivot_price) / pivot_price) * 100

                    if final_return >= threshold_pct * 0.5:  # Half threshold
                        new_outcome = "success"
                    elif final_return <= -threshold_pct * 0.5:
                        new_outcome = "failure"
                    else:
                        new_outcome = "neutral"

        # Track stats
        stats[new_outcome] = stats.get(new_outcome, 0) + 1

        if new_outcome != original_outcome:
            stats["changed"] += 1
            updates.append((new_outcome, row["id"]))
        else:
            stats["unchanged"] += 1

    # Apply updates
    if not dry_run and updates:
        logger.info(f"Updating {len(updates)} patterns...")
        with get_cursor() as cur:
            cur.executemany(
                "UPDATE detected_patterns SET outcome = ? WHERE id = ?",
                updates,
            )
        logger.info("Database updated successfully")
    elif dry_run:
        logger.info("DRY RUN - no database changes made")

    conn.close()

    # Print summary
    print("\n" + "=" * 60)
    print(f"RELABELING RESULTS (threshold: +/-{threshold_pct}%)")
    print("=" * 60)
    print(f"\nOriginal outcomes:")
    for outcome, count in sorted(original_counts.items()):
        pct = count / len(df) * 100
        print(f"  {outcome}: {count} ({pct:.1f}%)")

    print(f"\nNew outcomes:")
    for outcome in ["success", "failure", "neutral", "pending"]:
        count = stats.get(outcome, 0)
        pct = count / len(df) * 100 if len(df) > 0 else 0
        print(f"  {outcome}: {count} ({pct:.1f}%)")

    print(f"\nChanges:")
    print(f"  Unchanged: {stats['unchanged']}")
    print(f"  Changed: {stats['changed']}")

    # Calculate success rate
    total_labeled = stats["success"] + stats["failure"]
    if total_labeled > 0:
        success_rate = stats["success"] / total_labeled * 100
        print(f"\nNew success rate: {success_rate:.1f}% (was ~23%)")

    print("=" * 60)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Relabel patterns with symmetric thresholds")
    parser.add_argument(
        "--threshold",
        type=float,
        default=10.0,
        help="Symmetric threshold percentage (default: 10)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=8,
        help="Outcome window in weeks (default: 8)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show stats without updating database",
    )
    args = parser.parse_args()

    stats = relabel_patterns_symmetric(
        threshold_pct=args.threshold,
        outcome_window_weeks=args.window,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
