"""Extract CAN SLIM fundamental features.

Features:
- Original: eps_latest_yoy_growth, eps_acceleration, revenue_latest_yoy_growth,
            institutional_pct, market_cap_log
- Expanded: pe_ratio, peg_ratio, price_to_book, roe, roa, profit_margin,
            operating_margin, gross_margin, debt_to_equity, current_ratio,
            short_percent, analyst_rating, insider_pct
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

    # Start with default features
    features = _empty_features()
    features["institutional_pct"] = float(institutional_pct)
    features["market_cap_log"] = float(market_cap_log)

    # If no fundamental data, return defaults
    if len(fundamentals) < 1:
        return features

    # Latest quarter data
    latest = fundamentals.iloc[0]

    # EPS YoY growth
    eps_latest_yoy = latest.get("eps_yoy_growth")
    if eps_latest_yoy is not None and not pd.isna(eps_latest_yoy):
        features["eps_latest_yoy_growth"] = float(eps_latest_yoy)

    # EPS acceleration (current quarter growth - prior quarter growth)
    if len(fundamentals) >= 2:
        prior = fundamentals.iloc[1]
        prior_eps_yoy = prior.get("eps_yoy_growth")
        if prior_eps_yoy is not None and not pd.isna(prior_eps_yoy):
            if features["eps_latest_yoy_growth"] != 0.0:
                features["eps_acceleration"] = features["eps_latest_yoy_growth"] - prior_eps_yoy

    # Revenue YoY growth
    revenue_latest_yoy = latest.get("revenue_yoy_growth")
    if revenue_latest_yoy is not None and not pd.isna(revenue_latest_yoy):
        features["revenue_latest_yoy_growth"] = float(revenue_latest_yoy)

    # Expanded fundamentals (from most recent quarter with data)
    _extract_expanded_features(latest, features)

    return features


def _extract_expanded_features(row: pd.Series, features: dict) -> None:
    """Extract expanded fundamental features from a DataFrame row.

    Updates features dict in place.
    """
    # Valuation metrics
    pe = row.get("pe_ratio")
    if pe is not None and not pd.isna(pe) and pe > 0:
        features["pe_ratio"] = float(pe)
        # Normalize P/E to score (lower is better, capped at 50)
        features["pe_score"] = max(0, 1 - (min(pe, 50) / 50))

    peg = row.get("peg_ratio")
    if peg is not None and not pd.isna(peg) and peg > 0:
        features["peg_ratio"] = float(peg)
        # PEG < 1 is good, > 2 is expensive
        features["peg_score"] = max(0, 1 - (min(peg, 3) / 3))

    pb = row.get("price_to_book")
    if pb is not None and not pd.isna(pb) and pb > 0:
        features["price_to_book"] = float(pb)

    # Profitability metrics (already in decimal form from yfinance)
    roe = row.get("roe")
    if roe is not None and not pd.isna(roe):
        features["roe"] = float(roe)
        # Good ROE is > 15%, excellent is > 25%
        features["roe_score"] = min(1, max(0, roe / 0.30))

    roa = row.get("roa")
    if roa is not None and not pd.isna(roa):
        features["roa"] = float(roa)

    profit_margin = row.get("profit_margin")
    if profit_margin is not None and not pd.isna(profit_margin):
        features["profit_margin"] = float(profit_margin)
        # Good profit margin is > 10%, excellent is > 20%
        features["profit_margin_score"] = min(1, max(0, profit_margin / 0.25))

    op_margin = row.get("operating_margin")
    if op_margin is not None and not pd.isna(op_margin):
        features["operating_margin"] = float(op_margin)

    gross_margin = row.get("gross_margin")
    if gross_margin is not None and not pd.isna(gross_margin):
        features["gross_margin"] = float(gross_margin)

    # Financial health
    debt_equity = row.get("debt_to_equity")
    if debt_equity is not None and not pd.isna(debt_equity):
        features["debt_to_equity"] = float(debt_equity)
        # Low debt is good (< 50%), high debt (> 200%) is risky
        features["debt_score"] = max(0, 1 - (min(debt_equity, 200) / 200))

    current_ratio = row.get("current_ratio")
    if current_ratio is not None and not pd.isna(current_ratio):
        features["current_ratio"] = float(current_ratio)

    # Market sentiment
    short_pct = row.get("short_percent")
    if short_pct is not None and not pd.isna(short_pct):
        features["short_percent"] = float(short_pct)
        # Low short interest is good (< 5%), high (> 20%) is bearish
        features["short_score"] = max(0, 1 - (min(short_pct, 0.30) / 0.30))

    target = row.get("target_price")
    if target is not None and not pd.isna(target):
        features["analyst_target_price"] = float(target)

    rating = row.get("analyst_rating")
    if rating is not None and not pd.isna(rating):
        features["analyst_rating"] = float(rating)
        # Normalize to 0-1 (1=sell, 5=strong buy)
        features["analyst_rating_score"] = (rating - 1) / 4

    insider = row.get("insider_pct")
    if insider is not None and not pd.isna(insider):
        features["insider_pct"] = float(insider)


def _empty_features() -> dict:
    """Return dict of empty/default feature values."""
    return {
        # Original features
        "eps_latest_yoy_growth": 0.0,
        "eps_acceleration": 0.0,
        "revenue_latest_yoy_growth": 0.0,
        "institutional_pct": 0.5,
        "market_cap_log": 0.0,
        # Expanded features - valuation
        "pe_ratio": None,
        "pe_score": 0.5,
        "peg_ratio": None,
        "peg_score": 0.5,
        "price_to_book": None,
        # Profitability
        "roe": None,
        "roe_score": 0.5,
        "roa": None,
        "profit_margin": None,
        "profit_margin_score": 0.5,
        "operating_margin": None,
        "gross_margin": None,
        # Financial health
        "debt_to_equity": None,
        "debt_score": 0.5,
        "current_ratio": None,
        # Market sentiment
        "short_percent": None,
        "short_score": 0.5,
        "analyst_target_price": None,
        "analyst_rating": None,
        "analyst_rating_score": 0.5,
        "insider_pct": None,
    }
