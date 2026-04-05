#!/usr/bin/env python3
"""Initial data backfill script.

Pulls 10 years of daily OHLCV + fundamentals for the full stock universe.
Handles rate limiting, retries, and progress tracking. Resume-capable.

Usage:
    python scripts/backfill.py
    python scripts/backfill.py --test 10  # Test with 10 stocks first
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scanner.config import get
from scanner.db import init_db
from scanner.universe import refresh_universe, get_universe
from scanner.data_pipeline import (
    fetch_ohlcv,
    save_ohlcv_to_db,
    fetch_fundamentals,
    save_fundamentals_to_db,
    fetch_index_data,
    save_index_data_to_db,
    get_last_date_for_symbol,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def backfill(test_count: int = 0, skip_universe: bool = False, source: str = None) -> None:
    """Run the full backfill process.

    Args:
        test_count: If > 0, only process this many stocks (for testing).
        skip_universe: If True, don't refresh universe (use existing).
        source: Universe source override (sp500, nasdaq, nyse, all).
    """
    # Initialize database
    logger.info("Initializing database...")
    init_db()

    # Test mode: use a small set of well-known stocks (skip slow universe refresh)
    if test_count > 0:
        test_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "UNH",
                        "JNJ", "XOM", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "PEP", "KO"]
        symbols = test_symbols[:test_count]
        logger.info("Test mode: using %d test symbols: %s", len(symbols), symbols)

        # Save test symbols to database
        from scanner.db import get_cursor
        with get_cursor() as cur:
            for sym in symbols:
                cur.execute(
                    "INSERT OR IGNORE INTO stocks (symbol, last_updated) VALUES (?, date('now'))",
                    (sym,)
                )
    else:
        # Full mode: refresh entire universe
        if skip_universe:
            logger.info("Using existing universe...")
        else:
            logger.info("Refreshing stock universe (source: %s)...", source or "config default")
            count = refresh_universe(source=source)
            logger.info("Universe contains %d stocks", count)

        # Get symbols to process
        symbols = get_universe()
        if not symbols:
            logger.error("No symbols in universe. Run refresh_universe first.")
            return

    # Find symbols that need data (resume capability)
    symbols_to_fetch = []
    for symbol in symbols:
        last_date = get_last_date_for_symbol(symbol)
        if last_date is None:
            symbols_to_fetch.append(symbol)

    logger.info(
        "%d/%d symbols need data (resume: skipping %d already loaded)",
        len(symbols_to_fetch),
        len(symbols),
        len(symbols) - len(symbols_to_fetch),
    )

    # Fetch OHLCV data
    if symbols_to_fetch:
        logger.info("Fetching OHLCV data for %d symbols...", len(symbols_to_fetch))
        ohlcv_data = fetch_ohlcv(symbols_to_fetch)
        rows = save_ohlcv_to_db(ohlcv_data)
        logger.info("Saved %d OHLCV rows", rows)
    else:
        logger.info("All symbols already have OHLCV data")

    # Fetch fundamentals
    logger.info("Fetching fundamentals for %d symbols...", len(symbols))
    fundamentals = fetch_fundamentals(symbols)
    rows = save_fundamentals_to_db(fundamentals)
    logger.info("Saved %d fundamentals rows", rows)

    # Fetch index data
    rs_benchmark = get("features.rs_benchmark", "^GSPC")
    logger.info("Fetching index data for %s...", rs_benchmark)
    index_df = fetch_index_data(rs_benchmark)
    rows = save_index_data_to_db(rs_benchmark, index_df)
    logger.info("Saved %d index rows", rows)

    logger.info("Backfill complete!")


def main():
    parser = argparse.ArgumentParser(description="Backfill historical data")
    parser.add_argument(
        "--test",
        type=int,
        default=0,
        metavar="N",
        help="Test mode: only process N stocks",
    )
    parser.add_argument(
        "--skip-universe",
        action="store_true",
        help="Skip universe refresh, use existing stocks",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["sp500", "nasdaq", "nyse", "all"],
        default=None,
        help="Universe source: sp500, nasdaq, nyse, or all (default: from config)",
    )
    args = parser.parse_args()

    backfill(test_count=args.test, skip_universe=args.skip_universe, source=args.source)


if __name__ == "__main__":
    main()
