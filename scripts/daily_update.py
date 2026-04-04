#!/usr/bin/env python3
"""Daily incremental update script.

Fetches the latest OHLCV data for all stocks in the universe.
Run this daily to keep data current.

Usage:
    python scripts/daily_update.py
    python scripts/daily_update.py --days 5  # Fetch last 5 days
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scanner.config import get
from scanner.db import init_db
from scanner.universe import get_universe
from scanner.data_pipeline import (
    fetch_ohlcv,
    save_ohlcv_to_db,
    fetch_index_data,
    save_index_data_to_db,
    get_last_date_for_symbol,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def daily_update(days: int = 5) -> None:
    """Run daily incremental update.

    Args:
        days: Number of days to fetch (default 5 to cover weekends).
    """
    init_db()

    # Get symbols
    symbols = get_universe()
    if not symbols:
        logger.error("No symbols in universe. Run backfill.py first.")
        return

    logger.info("Updating %d symbols...", len(symbols))

    # Calculate date range
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    # Fetch and save OHLCV
    logger.info("Fetching OHLCV from %s to %s...", start_date, end_date)
    ohlcv_data = fetch_ohlcv(symbols, start_date=start_date, end_date=end_date)
    rows = save_ohlcv_to_db(ohlcv_data)
    logger.info("Updated %d OHLCV rows", rows)

    # Update index data
    rs_benchmark = get("features.rs_benchmark", "^GSPC")
    logger.info("Updating index data for %s...", rs_benchmark)
    index_df = fetch_index_data(rs_benchmark, start_date=start_date, end_date=end_date)
    rows = save_index_data_to_db(rs_benchmark, index_df)
    logger.info("Updated %d index rows", rows)

    # Report on data freshness
    sample_symbols = symbols[:5]
    for symbol in sample_symbols:
        last_date = get_last_date_for_symbol(symbol)
        logger.info("Latest data for %s: %s", symbol, last_date)

    logger.info("Daily update complete!")


def main():
    parser = argparse.ArgumentParser(description="Daily incremental data update")
    parser.add_argument(
        "--days",
        type=int,
        default=5,
        help="Number of days to fetch (default: 5)",
    )
    args = parser.parse_args()

    daily_update(days=args.days)


if __name__ == "__main__":
    main()
