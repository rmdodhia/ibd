"""Extract CAN SLIM fundamental features.

Features: eps_latest_yoy_growth, eps_acceleration,
revenue_latest_yoy_growth, institutional_pct, market_cap_log.
"""

import math
from typing import Optional

import pandas as pd

from scanner.db import get_connection


def extract_fundamental_features(
    symbol: str,
    breakout_date: str,
) -> dict:
    """Extract fundamental features for a stock.

    Args:
        symbol: Stock ticker symbol.
        breakout_date: Date of breakout (YYYY-MM-DD).

    Returns:
        Dict of fundamental features.
    """
    # Get stock info
    conn = get_connection()
    try:
        stock_info = pd.read_sql_query(
            "SELECT * FROM stocks WHERE symbol = ?",
            conn,
            params=[symbol],
        )

        fundamentals = pd.read_sql_query(
            """
            SELECT * FROM fundamentals
            WHERE symbol = ? AND quarter_end <= ?
            ORDER BY quarter_end DESC
            LIMIT 8
            """,
            conn,
            params=[symbol, breakout_date],
        )
    finally:
        conn.close()

    # Stock info features
    if not stock_info.empty:
        row = stock_info.iloc[0]
        institutional_pct = row.get("institutional_pct")
        market_cap = row.get("market_cap")

        if institutional_pct is None:
            institutional_pct = 0.5
        if market_cap is None or market_cap <= 0:
            market_cap_log = 0.0
        else:
            market_cap_log = math.log10(market_cap)
    else:
        institutional_pct = 0.5
        market_cap_log = 0.0

    # Fundamentals
    if len(fundamentals) < 2:
        return {
            "eps_latest_yoy_growth": 0.0,
            "eps_acceleration": 0.0,
            "revenue_latest_yoy_growth": 0.0,
            "institutional_pct": float(institutional_pct),
            "market_cap_log": float(market_cap_log),
        }

    # Latest quarter EPS YoY growth
    latest = fundamentals.iloc[0]
    eps_latest_yoy = latest.get("eps_yoy_growth")
    if eps_latest_yoy is None or pd.isna(eps_latest_yoy):
        eps_latest_yoy = 0.0

    # EPS acceleration (current quarter growth - prior quarter growth)
    if len(fundamentals) >= 2:
        prior = fundamentals.iloc[1]
        prior_eps_yoy = prior.get("eps_yoy_growth")
        if prior_eps_yoy is not None and not pd.isna(prior_eps_yoy):
            eps_acceleration = eps_latest_yoy - prior_eps_yoy
        else:
            eps_acceleration = 0.0
    else:
        eps_acceleration = 0.0

    # Latest quarter revenue YoY growth
    revenue_latest_yoy = latest.get("revenue_yoy_growth")
    if revenue_latest_yoy is None or pd.isna(revenue_latest_yoy):
        revenue_latest_yoy = 0.0

    return {
        "eps_latest_yoy_growth": float(eps_latest_yoy),
        "eps_acceleration": float(eps_acceleration),
        "revenue_latest_yoy_growth": float(revenue_latest_yoy),
        "institutional_pct": float(institutional_pct),
        "market_cap_log": float(market_cap_log),
    }


def _empty_features() -> dict:
    """Return dict of empty/default feature values."""
    return {
        "eps_latest_yoy_growth": 0.0,
        "eps_acceleration": 0.0,
        "revenue_latest_yoy_growth": 0.0,
        "institutional_pct": 0.5,
        "market_cap_log": 0.0,
    }
