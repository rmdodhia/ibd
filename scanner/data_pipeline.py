"""Data pipeline for pulling OHLCV and fundamentals via yfinance.

Handles batching, rate limiting, retries, and incremental updates.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

from scanner.config import get
from scanner.db import get_cursor, get_connection

logger = logging.getLogger(__name__)


def fetch_ohlcv(
    symbols: list[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    batch_size: Optional[int] = None,
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV data for multiple symbols.

    Args:
        symbols: List of ticker symbols.
        start_date: Start date (YYYY-MM-DD). Defaults to 10 years ago.
        end_date: End date (YYYY-MM-DD). Defaults to today.
        batch_size: Symbols per batch. Defaults to config value.

    Returns:
        Dict mapping symbol to DataFrame with OHLCV columns.
    """
    if batch_size is None:
        batch_size = get("data.yfinance_batch_size", 50)
    sleep_seconds = get("data.yfinance_sleep_seconds", 1)

    if start_date is None:
        years = get("data.history_years", 10)
        start_date = (datetime.now() - timedelta(days=years * 365)).strftime("%Y-%m-%d")
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    results = {}
    failed = []

    for i in range(0, len(symbols), batch_size):
        batch = symbols[i : i + batch_size]
        batch_str = " ".join(batch)

        logger.info(
            "Fetching OHLCV batch %d-%d of %d symbols",
            i + 1,
            min(i + batch_size, len(symbols)),
            len(symbols),
        )

        try:
            data = yf.download(
                batch_str,
                start=start_date,
                end=end_date,
                group_by="ticker",
                auto_adjust=False,
                progress=False,
                threads=True,
            )

            # Handle single vs multiple symbols
            if len(batch) == 1:
                symbol = batch[0]
                if not data.empty:
                    results[symbol] = data
            else:
                for symbol in batch:
                    try:
                        if symbol in data.columns.get_level_values(0):
                            df = data[symbol].dropna(how="all")
                            if not df.empty:
                                results[symbol] = df
                    except Exception as e:
                        logger.warning("Error extracting %s from batch: %s", symbol, e)
                        failed.append(symbol)

        except Exception as e:
            logger.error("Batch download failed: %s", e)
            failed.extend(batch)

        if i + batch_size < len(symbols):
            time.sleep(sleep_seconds)

    if failed:
        logger.warning("Failed to fetch %d symbols: %s", len(failed), failed[:10])

    return results


def save_ohlcv_to_db(data: dict[str, pd.DataFrame]) -> int:
    """Save OHLCV data to database.

    Args:
        data: Dict mapping symbol to DataFrame.

    Returns:
        Number of rows inserted.
    """
    total_rows = 0
    conn = get_connection()
    try:
        cur = conn.cursor()

        for symbol, df in data.items():
            df = df.reset_index()
            # Normalize column names
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]

            if "date" not in df.columns:
                if "index" in df.columns:
                    df = df.rename(columns={"index": "date"})
                else:
                    continue

            for _, row in df.iterrows():
                try:
                    cur.execute(
                        """
                        INSERT OR REPLACE INTO daily_prices
                        (symbol, date, open, high, low, close, adj_close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            symbol,
                            row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else str(row["date"])[:10],
                            float(row.get("open", 0)),
                            float(row.get("high", 0)),
                            float(row.get("low", 0)),
                            float(row.get("close", 0)),
                            float(row.get("adj_close", row.get("close", 0))),
                            int(row.get("volume", 0)),
                        ),
                    )
                    total_rows += 1
                except Exception as e:
                    logger.warning("Error saving row for %s: %s", symbol, e)

        conn.commit()
    finally:
        conn.close()

    logger.info("Saved %d OHLCV rows to database", total_rows)
    return total_rows


def fetch_fundamentals(symbols: list[str]) -> dict[str, list[dict]]:
    """Fetch quarterly fundamentals (EPS, revenue) for symbols.

    Uses quarterly_income_stmt which provides Net Income and Total Revenue.

    Args:
        symbols: List of ticker symbols.

    Returns:
        Dict mapping symbol to list of quarterly data dicts.
    """
    sleep_seconds = get("data.yfinance_sleep_seconds", 1)
    results = {}

    for i, symbol in enumerate(symbols):
        if i > 0 and i % 50 == 0:
            logger.info("Fetching fundamentals: %d/%d", i, len(symbols))
            time.sleep(sleep_seconds)

        try:
            ticker = yf.Ticker(symbol)

            # Get quarterly income statement (newer API)
            income_stmt = ticker.quarterly_income_stmt
            if income_stmt is None or income_stmt.empty:
                continue

            # Get shares outstanding for EPS calculation
            shares = None
            try:
                info = ticker.info
                shares = info.get("sharesOutstanding")
            except Exception:
                pass

            quarters = []
            # Columns are dates, rows are line items
            for col in income_stmt.columns:
                quarter_end = col.strftime("%Y-%m-%d") if hasattr(col, "strftime") else str(col)[:10]

                # Extract Net Income and Total Revenue
                net_income = None
                revenue = None

                if "Net Income" in income_stmt.index:
                    val = income_stmt.loc["Net Income", col]
                    if pd.notna(val):
                        net_income = float(val)

                if "Total Revenue" in income_stmt.index:
                    val = income_stmt.loc["Total Revenue", col]
                    if pd.notna(val):
                        revenue = float(val)

                # Calculate EPS if we have net income and shares
                eps = None
                if net_income is not None and shares:
                    eps = net_income / shares

                if net_income is not None or revenue is not None:
                    quarters.append({
                        "quarter_end": quarter_end,
                        "eps": eps,
                        "revenue": revenue,
                    })

            if quarters:
                results[symbol] = quarters

        except Exception as e:
            logger.warning("Failed to fetch fundamentals for %s: %s", symbol, e)

    return results


def fetch_expanded_fundamentals(symbols: list[str]) -> dict[str, dict]:
    """Fetch expanded fundamentals (valuation, profitability, debt) for symbols.

    Gets snapshot data from yfinance info endpoint including:
    - Valuation: P/E, PEG, P/B
    - Profitability: ROE, ROA, margins
    - Financial health: debt/equity, current ratio
    - Market sentiment: short interest, analyst ratings

    Args:
        symbols: List of ticker symbols.

    Returns:
        Dict mapping symbol to dict of expanded fundamental data.
    """
    sleep_seconds = get("data.yfinance_sleep_seconds", 1)
    results = {}

    for i, symbol in enumerate(symbols):
        if i > 0 and i % 50 == 0:
            logger.info("Fetching expanded fundamentals: %d/%d", i, len(symbols))
            time.sleep(sleep_seconds)

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            if not info:
                continue

            # Extract expanded fundamentals
            data = {
                # Valuation metrics
                "pe_ratio": info.get("trailingPE") or info.get("forwardPE"),
                "peg_ratio": info.get("pegRatio"),
                "price_to_book": info.get("priceToBook"),

                # Profitability metrics
                "roe": info.get("returnOnEquity"),
                "roa": info.get("returnOnAssets"),
                "profit_margin": info.get("profitMargins"),
                "operating_margin": info.get("operatingMargins"),
                "gross_margin": info.get("grossMargins"),

                # Financial health
                "debt_to_equity": info.get("debtToEquity"),
                "current_ratio": info.get("currentRatio"),
                "quick_ratio": info.get("quickRatio"),
                "free_cash_flow": info.get("freeCashflow"),

                # Market sentiment
                "short_percent": info.get("shortPercentOfFloat"),
                "target_price": info.get("targetMeanPrice"),
                "analyst_rating": _convert_analyst_rating(info.get("recommendationKey")),
                "insider_pct": info.get("heldPercentInsiders"),
            }

            # Only store if we got some data
            if any(v is not None for v in data.values()):
                results[symbol] = data

        except Exception as e:
            logger.warning("Failed to fetch expanded fundamentals for %s: %s", symbol, e)

    return results


def _convert_analyst_rating(rating_key: Optional[str]) -> Optional[float]:
    """Convert analyst rating key to numeric scale (1-5, 5=strong buy).

    Args:
        rating_key: String rating from yfinance (e.g., "buy", "hold", "sell").

    Returns:
        Numeric rating 1-5 or None if unknown.
    """
    if not rating_key:
        return None

    rating_map = {
        "strongBuy": 5.0,
        "buy": 4.0,
        "hold": 3.0,
        "underperform": 2.0,
        "sell": 1.0,
        "strongSell": 1.0,
    }

    # Handle case variations
    key = rating_key.lower().replace("_", "").replace("-", "").replace(" ", "")
    for k, v in rating_map.items():
        if k.lower() in key or key in k.lower():
            return v

    return None


def save_expanded_fundamentals_to_db(data: dict[str, dict]) -> int:
    """Save expanded fundamentals to database.

    Updates the most recent quarter for each symbol with the snapshot data.

    Args:
        data: Dict mapping symbol to expanded fundamental data.

    Returns:
        Number of rows updated.
    """
    total_rows = 0

    with get_cursor() as cur:
        for symbol, fundamentals in data.items():
            # Find the most recent quarter for this symbol
            cur.execute(
                "SELECT quarter_end FROM fundamentals WHERE symbol = ? ORDER BY quarter_end DESC LIMIT 1",
                (symbol,),
            )
            row = cur.fetchone()

            if row:
                quarter_end = row[0]
            else:
                # No existing quarter, skip (need quarterly data first)
                continue

            try:
                cur.execute(
                    """
                    UPDATE fundamentals SET
                        pe_ratio = ?,
                        peg_ratio = ?,
                        price_to_book = ?,
                        roe = ?,
                        roa = ?,
                        profit_margin = ?,
                        operating_margin = ?,
                        gross_margin = ?,
                        debt_to_equity = ?,
                        current_ratio = ?,
                        quick_ratio = ?,
                        free_cash_flow = ?,
                        short_percent = ?,
                        target_price = ?,
                        analyst_rating = ?,
                        insider_pct = ?
                    WHERE symbol = ? AND quarter_end = ?
                    """,
                    (
                        fundamentals.get("pe_ratio"),
                        fundamentals.get("peg_ratio"),
                        fundamentals.get("price_to_book"),
                        fundamentals.get("roe"),
                        fundamentals.get("roa"),
                        fundamentals.get("profit_margin"),
                        fundamentals.get("operating_margin"),
                        fundamentals.get("gross_margin"),
                        fundamentals.get("debt_to_equity"),
                        fundamentals.get("current_ratio"),
                        fundamentals.get("quick_ratio"),
                        fundamentals.get("free_cash_flow"),
                        fundamentals.get("short_percent"),
                        fundamentals.get("target_price"),
                        fundamentals.get("analyst_rating"),
                        fundamentals.get("insider_pct"),
                        symbol,
                        quarter_end,
                    ),
                )
                total_rows += 1
            except Exception as e:
                logger.warning("Error saving expanded fundamentals for %s: %s", symbol, e)

    logger.info("Updated %d rows with expanded fundamentals", total_rows)
    return total_rows


def save_fundamentals_to_db(data: dict[str, list[dict]]) -> int:
    """Save fundamentals data to database with YoY growth calculation.

    Args:
        data: Dict mapping symbol to list of quarterly data.

    Returns:
        Number of rows inserted.
    """
    total_rows = 0

    with get_cursor() as cur:
        for symbol, quarters in data.items():
            # Sort by quarter date
            quarters = sorted(quarters, key=lambda x: x["quarter_end"])

            for i, q in enumerate(quarters):
                eps = q.get("eps")
                revenue = q.get("revenue")

                # Calculate YoY growth if we have data from 4 quarters ago
                eps_yoy = None
                revenue_yoy = None

                if i >= 4:
                    prev = quarters[i - 4]
                    if eps is not None and prev.get("eps") and prev["eps"] != 0:
                        eps_yoy = ((eps - prev["eps"]) / abs(prev["eps"])) * 100
                    if revenue is not None and prev.get("revenue") and prev["revenue"] != 0:
                        revenue_yoy = ((revenue - prev["revenue"]) / abs(prev["revenue"])) * 100

                try:
                    cur.execute(
                        """
                        INSERT OR REPLACE INTO fundamentals
                        (symbol, quarter_end, eps, revenue, eps_yoy_growth, revenue_yoy_growth)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (symbol, q["quarter_end"], eps, revenue, eps_yoy, revenue_yoy),
                    )
                    total_rows += 1
                except Exception as e:
                    logger.warning("Error saving fundamentals for %s: %s", symbol, e)

    logger.info("Saved %d fundamentals rows to database", total_rows)
    return total_rows


def fetch_index_data(
    symbol: str = "^GSPC",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch index data (S&P 500 by default).

    Args:
        symbol: Index symbol. Defaults to ^GSPC (S&P 500).
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).

    Returns:
        DataFrame with index OHLCV.
    """
    if start_date is None:
        years = get("data.history_years", 10)
        start_date = (datetime.now() - timedelta(days=years * 365)).strftime("%Y-%m-%d")
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    logger.info("Fetching index data for %s", symbol)

    data = yf.download(
        symbol,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
    )

    return data


def save_index_data_to_db(symbol: str, df: pd.DataFrame) -> int:
    """Save index data to database.

    Args:
        symbol: Index symbol.
        df: DataFrame with index data.

    Returns:
        Number of rows inserted.
    """
    total_rows = 0
    df = df.reset_index()

    # Handle MultiIndex columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]

    with get_cursor() as cur:
        for _, row in df.iterrows():
            try:
                date_str = row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else str(row["date"])[:10]
                cur.execute(
                    """
                    INSERT OR REPLACE INTO index_prices
                    (symbol, date, close, volume)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        symbol,
                        date_str,
                        float(row.get("close", 0)),
                        int(row.get("volume", 0)),
                    ),
                )
                total_rows += 1
            except Exception as e:
                logger.warning("Error saving index row: %s", e)

    logger.info("Saved %d index rows for %s", total_rows, symbol)
    return total_rows


def get_last_date_for_symbol(symbol: str) -> Optional[str]:
    """Get the most recent date we have data for a symbol.

    Args:
        symbol: Ticker symbol.

    Returns:
        Date string (YYYY-MM-DD) or None if no data.
    """
    with get_cursor() as cur:
        cur.execute(
            "SELECT MAX(date) FROM daily_prices WHERE symbol = ?",
            (symbol,),
        )
        row = cur.fetchone()
        return row[0] if row and row[0] else None


def get_price_data(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Load price data for a symbol from database.

    Args:
        symbol: Ticker symbol.
        start_date: Optional start date filter.
        end_date: Optional end date filter.

    Returns:
        DataFrame with columns [date, open, high, low, close, adj_close, volume].
    """
    query = "SELECT * FROM daily_prices WHERE symbol = ?"
    params = [symbol]

    if start_date:
        query += " AND date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND date <= ?"
        params.append(end_date)

    query += " ORDER BY date ASC"

    conn = get_connection()
    try:
        df = pd.read_sql_query(query, conn, params=params)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df
    finally:
        conn.close()


def get_index_data(
    symbol: str = "^GSPC",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Load index data from database.

    Args:
        symbol: Index symbol.
        start_date: Optional start date filter.
        end_date: Optional end date filter.

    Returns:
        DataFrame with columns [date, close, volume].
    """
    query = "SELECT * FROM index_prices WHERE symbol = ?"
    params = [symbol]

    if start_date:
        query += " AND date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND date <= ?"
        params.append(end_date)

    query += " ORDER BY date ASC"

    conn = get_connection()
    try:
        df = pd.read_sql_query(query, conn, params=params)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df
    finally:
        conn.close()
