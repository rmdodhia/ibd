"""Stock universe management.

Pull S&P 500 symbols, apply filters (price, volume, market cap),
and maintain the universe list.
"""

import logging
import time
from typing import Optional

import pandas as pd
import yfinance as yf

from scanner.config import get
from scanner.db import get_cursor, init_db

logger = logging.getLogger(__name__)


def get_sp500_symbols() -> list[str]:
    """Fetch current S&P 500 constituents from Wikipedia.

    Returns:
        List of ticker symbols.
    """
    import requests
    from io import StringIO

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))
    df = tables[0]
    # Symbol column contains tickers, some have dots (BRK.B) that need conversion
    symbols = df["Symbol"].str.replace(".", "-", regex=False).tolist()
    logger.info("Fetched %d S&P 500 symbols", len(symbols))
    return symbols


def get_stock_info(symbols: list[str], batch_size: int = 50) -> list[dict]:
    """Fetch stock metadata for filtering.

    Args:
        symbols: List of ticker symbols.
        batch_size: Number of symbols per batch (for rate limiting).

    Returns:
        List of dicts with symbol info (symbol, name, sector, market_cap, etc.).
    """
    sleep_seconds = get("data.yfinance_sleep_seconds", 1)
    results = []

    for i in range(0, len(symbols), batch_size):
        batch = symbols[i : i + batch_size]
        logger.info("Fetching info for batch %d-%d of %d", i, i + len(batch), len(symbols))

        for symbol in batch:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                results.append({
                    "symbol": symbol,
                    "name": info.get("shortName") or info.get("longName"),
                    "sector": info.get("sector"),
                    "industry": info.get("industry"),
                    "market_cap": info.get("marketCap"),
                    "shares_outstanding": info.get("sharesOutstanding"),
                    "float_shares": info.get("floatShares"),
                    "institutional_pct": info.get("heldPercentInstitutions"),
                })
            except Exception as e:
                logger.warning("Failed to fetch info for %s: %s", symbol, e)
                continue

        if i + batch_size < len(symbols):
            time.sleep(sleep_seconds)

    return results


def filter_universe(stocks: list[dict]) -> list[dict]:
    """Apply price, volume, and market cap filters.

    Args:
        stocks: List of stock info dicts.

    Returns:
        Filtered list of stocks meeting all criteria.
    """
    min_price = get("universe.min_price", 10)
    min_volume = get("universe.min_avg_volume", 200000)
    min_market_cap = get("universe.min_market_cap", 500_000_000)

    filtered = []
    for stock in stocks:
        market_cap = stock.get("market_cap")
        if market_cap is None or market_cap < min_market_cap:
            continue
        filtered.append(stock)

    logger.info(
        "Filtered to %d stocks (market cap > $%s)",
        len(filtered),
        f"{min_market_cap:,.0f}",
    )
    return filtered


def save_stocks_to_db(stocks: list[dict]) -> int:
    """Save stock metadata to database.

    Args:
        stocks: List of stock info dicts.

    Returns:
        Number of stocks saved.
    """
    with get_cursor() as cur:
        for stock in stocks:
            cur.execute(
                """
                INSERT OR REPLACE INTO stocks
                (symbol, name, sector, industry, market_cap, shares_outstanding,
                 float_shares, institutional_pct, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, date('now'))
                """,
                (
                    stock["symbol"],
                    stock.get("name"),
                    stock.get("sector"),
                    stock.get("industry"),
                    stock.get("market_cap"),
                    stock.get("shares_outstanding"),
                    stock.get("float_shares"),
                    stock.get("institutional_pct"),
                ),
            )
    logger.info("Saved %d stocks to database", len(stocks))
    return len(stocks)


def get_universe() -> list[str]:
    """Get list of symbols currently in the universe.

    Returns:
        List of ticker symbols from database.
    """
    with get_cursor() as cur:
        cur.execute("SELECT symbol FROM stocks ORDER BY symbol")
        return [row[0] for row in cur.fetchall()]


def refresh_universe() -> int:
    """Refresh the stock universe from scratch.

    Pulls S&P 500 symbols, fetches info, filters, and saves to DB.

    Returns:
        Number of stocks in the refreshed universe.
    """
    init_db()

    source = get("universe.source", "sp500")
    if source != "sp500":
        raise ValueError(f"Unknown universe source: {source}")

    symbols = get_sp500_symbols()
    stocks = get_stock_info(symbols)
    filtered = filter_universe(stocks)
    count = save_stocks_to_db(filtered)

    return count


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    count = refresh_universe()
    print(f"Universe refreshed: {count} stocks")
