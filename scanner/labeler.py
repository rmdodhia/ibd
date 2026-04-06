"""Historical breakout detection and auto-labeling.

Scans each stock's history to find breakout events,
classifies the preceding pattern, and labels outcomes.

Optimizations:
- Multiprocessing for parallel symbol processing
- Batch database writes
- Efficient DataFrame operations

Usage:
    python -m scanner.labeler
    python -m scanner.labeler --symbol AAPL
    python -m scanner.labeler --limit 100
    python -m scanner.labeler --workers 8
"""

import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import pandas as pd
from tqdm import tqdm

from scanner.config import get
from scanner.db import get_cursor, get_connection, init_db
from scanner.universe import get_universe
from scanner.data_pipeline import get_price_data, get_index_data
from scanner.breakout_detector import (
    detect_breakouts,
    label_breakout_outcomes,
    Breakout,
)
from scanner.patterns import classify_pattern
from scanner.features import extract_all_features
from scanner.quality_score import compute_quality_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_optimal_workers() -> int:
    """Get optimal number of parallel workers."""
    cpu_count = os.cpu_count() or 1
    # Use 75% of CPUs, max 12, min 1
    return min(max(cpu_count * 3 // 4, 1), 12)


def run_labeler(
    symbol: Optional[str] = None,
    limit: Optional[int] = None,
    skip_existing: bool = True,
    num_workers: Optional[int] = None,
) -> dict:
    """Run the historical labeling process.

    Args:
        symbol: Optional single symbol to process.
        limit: Optional limit on number of symbols to process.
        skip_existing: Skip symbols that already have labeled breakouts.
        num_workers: Number of parallel workers. None = auto-detect.

    Returns:
        Dict with stats (total_breakouts, success_count, failure_count).
    """
    init_db()

    # Get symbols to process
    if symbol:
        symbols = [symbol]
        num_workers = 1  # Single symbol, no parallelization needed
    else:
        symbols = get_universe()
        if not symbols:
            logger.error("No symbols in universe. Run backfill first.")
            return {"total_breakouts": 0}

    if limit:
        symbols = symbols[:limit]

    # Filter out already processed symbols
    if skip_existing and not symbol:
        with get_cursor() as cur:
            cur.execute("SELECT DISTINCT symbol FROM detected_patterns")
            processed = {row[0] for row in cur.fetchall()}
        symbols = [s for s in symbols if s not in processed]
        logger.info("Skipping %d already processed symbols", len(processed))

    if not symbols:
        logger.info("No symbols to process")
        return {"total_breakouts": 0, "symbols_processed": 0}

    # Get index data for RS calculations (shared across workers)
    rs_benchmark = get("features.rs_benchmark", "^GSPC")
    index_df = get_index_data(rs_benchmark)
    if index_df.empty:
        logger.warning("No index data available. RS features will be empty.")

    stats = {
        "total_breakouts": 0,
        "success_count": 0,
        "failure_count": 0,
        "neutral_count": 0,
        "pending_count": 0,
        "symbols_processed": 0,
    }

    logger.info("Processing %d symbols...", len(symbols))

    # Determine number of workers
    if num_workers is None:
        num_workers = get_optimal_workers()

    if num_workers > 1 and len(symbols) > 1:
        # Parallel processing
        logger.info("Using %d parallel workers", num_workers)
        stats = _process_parallel(symbols, index_df, num_workers)
    else:
        # Sequential processing (for single symbol or debugging)
        stats = _process_sequential(symbols, index_df)

    logger.info(
        "Labeling complete. Total: %d breakouts, %d success, %d failure, %d neutral",
        stats["total_breakouts"],
        stats["success_count"],
        stats["failure_count"],
        stats["neutral_count"],
    )

    return stats


def _process_sequential(symbols: list[str], index_df: pd.DataFrame) -> dict:
    """Process symbols sequentially (for debugging or single symbol)."""
    stats = {
        "total_breakouts": 0,
        "success_count": 0,
        "failure_count": 0,
        "neutral_count": 0,
        "pending_count": 0,
        "symbols_processed": 0,
    }

    for sym in tqdm(symbols, desc="Labeling"):
        try:
            result = _process_symbol(sym, index_df)
            if result:
                stats["total_breakouts"] += result["breakouts"]
                stats["success_count"] += result["success"]
                stats["failure_count"] += result["failure"]
                stats["neutral_count"] += result.get("neutral", 0)
                stats["pending_count"] += result.get("pending", 0)
                stats["symbols_processed"] += 1
        except Exception as e:
            logger.warning("Error processing %s: %s", sym, e)

    return stats


def _process_parallel(
    symbols: list[str],
    index_df: pd.DataFrame,
    num_workers: int,
) -> dict:
    """Process symbols in parallel using ProcessPoolExecutor."""
    stats = {
        "total_breakouts": 0,
        "success_count": 0,
        "failure_count": 0,
        "neutral_count": 0,
        "pending_count": 0,
        "symbols_processed": 0,
    }

    # Convert index_df to dict for serialization
    index_data = index_df.to_dict("list") if not index_df.empty else None

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(_process_symbol_worker, sym, index_data): sym
            for sym in symbols
        }

        # Collect results with progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc="Labeling"):
            sym = futures[future]
            try:
                result = future.result()
                if result:
                    stats["total_breakouts"] += result["breakouts"]
                    stats["success_count"] += result["success"]
                    stats["failure_count"] += result["failure"]
                    stats["neutral_count"] += result.get("neutral", 0)
                    stats["pending_count"] += result.get("pending", 0)
                    stats["symbols_processed"] += 1
            except Exception as e:
                logger.warning("Error processing %s: %s", sym, e)

    return stats


def _process_symbol_worker(symbol: str, index_data: Optional[dict]) -> Optional[dict]:
    """Worker function for parallel processing (must be picklable)."""
    # Reconstruct index DataFrame
    if index_data:
        index_df = pd.DataFrame(index_data)
    else:
        index_df = pd.DataFrame()

    return _process_symbol(symbol, index_df)


def _process_symbol(
    symbol: str,
    index_df: pd.DataFrame,
) -> Optional[dict]:
    """Process a single symbol: detect breakouts, classify patterns, extract features.

    Returns:
        Dict with counts or None if skipped.
    """
    # Get price data
    stock_df = get_price_data(symbol)
    if stock_df.empty or len(stock_df) < 100:
        return None

    # Detect breakouts
    breakouts = detect_breakouts(stock_df, symbol)
    if not breakouts:
        return None

    # Label outcomes
    breakouts = label_breakout_outcomes(breakouts, stock_df)

    # Classify patterns and extract features (batch for efficiency)
    _classify_and_save_batch(breakouts, stock_df, index_df)

    # Count outcomes
    result = {
        "breakouts": len(breakouts),
        "success": sum(1 for b in breakouts if b.outcome == "success"),
        "failure": sum(1 for b in breakouts if b.outcome == "failure"),
        "neutral": sum(1 for b in breakouts if b.outcome == "neutral"),
        "pending": sum(1 for b in breakouts if b.outcome == "pending"),
    }

    return result


def _classify_and_save_batch(
    breakouts: list[Breakout],
    stock_df: pd.DataFrame,
    index_df: pd.DataFrame,
) -> None:
    """Classify patterns and save all breakouts in a batch.

    More efficient than individual inserts.
    """
    # Prepare batch data
    pattern_rows = []
    feature_rows = []

    for breakout in breakouts:
        # Classify the pattern
        pattern_type, confidence, metadata = classify_pattern(
            stock_df, breakout.symbol, breakout.breakout_date
        )

        pattern_rows.append((
            breakout.symbol,
            pattern_type,
            breakout.consolidation_start,
            breakout.breakout_date,
            breakout.breakout_date,
            breakout.breakout_price,
            breakout.outcome,
            breakout.outcome_return_pct,
            breakout.max_gain_pct,
            breakout.max_loss_pct,
            # Multi-label outcomes
            breakout.outcome_asym_20_7,
            breakout.outcome_asym_15_10,
            breakout.outcome_sym_10,
            breakout.return_asym_20_7,
            breakout.return_asym_15_10,
            breakout.return_sym_10,
            "auto",
        ))

        # Extract features
        features = extract_all_features(
            symbol=breakout.symbol,
            stock_df=stock_df,
            index_df=index_df,
            base_start_date=breakout.consolidation_start,
            base_end_date=breakout.breakout_date,
            breakout_date=breakout.breakout_date,
            pattern_metadata=metadata,
        )

        # Compute quality score
        try:
            quality = compute_quality_score(
                stock_df=stock_df,
                index_df=index_df,
                features=features,
                base_start_date=breakout.consolidation_start,
                base_end_date=breakout.breakout_date,
            )
            features["quality_score"] = quality.total_score
            features["technical_score"] = quality.technical_score
            features["fundamental_score"] = quality.fundamental_score
            features["market_score"] = quality.market_score
            features["prior_uptrend_pct"] = quality.prior_uptrend_pct
        except Exception as e:
            logger.debug("Could not compute quality score: %s", e)
            features["quality_score"] = 0.0
            features["technical_score"] = 0.0
            features["fundamental_score"] = 0.0
            features["market_score"] = 0.0
            features["prior_uptrend_pct"] = 0.0

        feature_rows.append(features)

    # Batch insert patterns
    with get_cursor() as cur:
        cur.executemany(
            """
            INSERT INTO detected_patterns
            (symbol, pattern_type, base_start_date, base_end_date,
             pivot_date, pivot_price, outcome, outcome_return_pct,
             outcome_max_gain_pct, outcome_max_loss_pct,
             outcome_asym_20_7, outcome_asym_15_10, outcome_sym_10,
             return_asym_20_7, return_asym_15_10, return_sym_10,
             auto_label)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            pattern_rows,
        )

        # Get the inserted IDs
        cur.execute(
            "SELECT id FROM detected_patterns ORDER BY id DESC LIMIT ?",
            (len(pattern_rows),),
        )
        pattern_ids = [row[0] for row in cur.fetchall()][::-1]

        # Batch insert features
        for pattern_id, features in zip(pattern_ids, feature_rows):
            cur.execute(
                """
                INSERT OR REPLACE INTO pattern_features
                (pattern_id, base_depth_pct, base_duration_weeks, base_symmetry,
                 handle_depth_pct, tightness_score, pre_breakout_tightness, pre_breakout_range_pct,
                 breakout_volume_ratio, volume_trend_in_base, up_down_volume_ratio,
                 rs_line_slope_4wk, rs_line_slope_12wk, rs_acceleration, rs_new_high, rs_rank_percentile,
                 eps_latest_yoy_growth, eps_acceleration, revenue_latest_yoy_growth,
                 institutional_pct, market_cap_log,
                 sp500_above_200dma, sp500_trend_4wk, price_vs_50dma, price_vs_200dma,
                 quality_score, technical_score, fundamental_score, market_score, prior_uptrend_pct)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pattern_id,
                    features.get("base_depth_pct", 0),
                    features.get("base_duration_weeks", 0),
                    features.get("base_symmetry", 0.5),
                    features.get("handle_depth_pct", 0),
                    features.get("tightness_score", 0.5),
                    features.get("pre_breakout_tightness", 1.0),
                    features.get("pre_breakout_range_pct", 0.0),
                    features.get("breakout_volume_ratio", 1),
                    features.get("volume_trend_in_base", 0),
                    features.get("up_down_volume_ratio", 1),
                    features.get("rs_line_slope_4wk", 0),
                    features.get("rs_line_slope_12wk", 0),
                    features.get("rs_acceleration", 0.0),
                    features.get("rs_new_high", False),
                    features.get("rs_rank_percentile", 50),
                    features.get("eps_latest_yoy_growth", 0),
                    features.get("eps_acceleration", 0),
                    features.get("revenue_latest_yoy_growth", 0),
                    features.get("institutional_pct", 0.5),
                    features.get("market_cap_log", 0),
                    features.get("sp500_above_200dma", True),
                    features.get("sp500_trend_4wk", 0),
                    features.get("price_vs_50dma", 0),
                    features.get("price_vs_200dma", 0),
                    features.get("quality_score", 0),
                    features.get("technical_score", 0),
                    features.get("fundamental_score", 0),
                    features.get("market_score", 0),
                    features.get("prior_uptrend_pct", 0),
                ),
            )


def get_labeled_data(label_strategy: str = None) -> pd.DataFrame:
    """Load all labeled patterns with features for training.

    Args:
        label_strategy: Which outcome column to use for filtering.
            Options: "asym_20_7", "asym_15_10", "sym_10"
            If None, uses legacy 'outcome' column.

    Returns:
        DataFrame with pattern info and features.
    """
    # Map label strategy to outcome column
    strategy_to_column = {
        "asym_20_7": "outcome_asym_20_7",
        "asym_15_10": "outcome_asym_15_10",
        "sym_10": "outcome_sym_10",
    }

    # Determine which outcome column to filter on
    if label_strategy and label_strategy in strategy_to_column:
        outcome_col = strategy_to_column[label_strategy]
    else:
        outcome_col = "outcome"

    conn = get_connection()
    try:
        # Check if new columns exist
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(detected_patterns)")
        columns = {row[1] for row in cur.fetchall()}

        # Fall back to legacy column if new columns don't exist
        if outcome_col not in columns:
            logger.warning(
                "Column %s not found, falling back to 'outcome'. "
                "Run 'python -m scanner.labeler --force' to populate new columns.",
                outcome_col
            )
            outcome_col = "outcome"

        df = pd.read_sql_query(
            f"""
            SELECT p.*, f.*
            FROM detected_patterns p
            LEFT JOIN pattern_features f ON p.id = f.pattern_id
            WHERE p.{outcome_col} IN ('success', 'failure')
            ORDER BY p.pivot_date
            """,
            conn,
        )

        # If using a specific strategy, add a normalized 'outcome' column for training
        if label_strategy and label_strategy in strategy_to_column and outcome_col in df.columns:
            df["outcome_for_training"] = df[outcome_col]
        else:
            df["outcome_for_training"] = df["outcome"]

        return df
    finally:
        conn.close()


def get_human_labeled_data() -> pd.DataFrame:
    """Load only human-reviewed patterns for evaluation.

    Returns patterns where a human has reviewed and labeled the outcome.
    Used as a gold standard test set to evaluate model performance
    and analyze disagreements between auto-labels and human judgment.

    Returns:
        DataFrame with pattern info and features for reviewed patterns.
    """
    conn = get_connection()
    try:
        df = pd.read_sql_query(
            """
            SELECT p.*, f.*
            FROM detected_patterns p
            LEFT JOIN pattern_features f ON p.id = f.pattern_id
            WHERE p.reviewed = 1
              AND p.human_label IN ('success', 'failure')
            ORDER BY p.pivot_date
            """,
            conn,
        )
        logger.info("Loaded %d human-labeled patterns", len(df))
        return df
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Run historical breakout labeling")
    parser.add_argument(
        "--symbol",
        type=str,
        help="Process a single symbol",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of symbols to process",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process symbols that already have labels",
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of parallel workers (default: auto)",
    )
    args = parser.parse_args()

    stats = run_labeler(
        symbol=args.symbol,
        limit=args.limit,
        skip_existing=not args.force,
        num_workers=args.workers,
    )

    print(f"\nLabeling Results:")
    print(f"  Symbols processed: {stats['symbols_processed']}")
    print(f"  Total breakouts: {stats['total_breakouts']}")
    print(f"  Successful: {stats['success_count']}")
    print(f"  Failed: {stats['failure_count']}")
    print(f"  Neutral: {stats['neutral_count']}")


if __name__ == "__main__":
    main()
