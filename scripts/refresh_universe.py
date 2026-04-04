#!/usr/bin/env python3
"""Refresh stock universe.

Re-pulls S&P 500 / NYSE+NASDAQ symbols, applies filters,
adds new stocks, flags delisted ones.

Usage:
    python scripts/refresh_universe.py
"""

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scanner.db import init_db
from scanner.universe import refresh_universe, get_universe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Refresh the stock universe."""
    init_db()

    # Get current universe for comparison
    old_symbols = set(get_universe())
    logger.info("Current universe: %d symbols", len(old_symbols))

    # Refresh
    count = refresh_universe()

    # Compare
    new_symbols = set(get_universe())
    added = new_symbols - old_symbols
    removed = old_symbols - new_symbols

    if added:
        logger.info("Added %d new symbols: %s", len(added), sorted(added)[:10])
    if removed:
        logger.info("Removed %d symbols: %s", len(removed), sorted(removed)[:10])

    logger.info("Universe refreshed: %d total symbols", count)


if __name__ == "__main__":
    main()
