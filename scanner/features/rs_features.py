"""Extract relative strength features.

Features: rs_line_slope_4wk, rs_line_slope_12wk,
rs_new_high, rs_rank_percentile.
"""

import numpy as np
import pandas as pd

from scanner.config import get


def extract_rs_features(
    stock_df: pd.DataFrame,
    index_df: pd.DataFrame,
    breakout_date: str,
) -> dict:
    """Extract relative strength features.

    Args:
        stock_df: Stock price DataFrame with [date, close].
        index_df: Index price DataFrame (S&P 500) with [date, close].
        breakout_date: Date of breakout (YYYY-MM-DD).

    Returns:
        Dict of RS features.
    """
    stock_df = stock_df.copy()
    index_df = index_df.copy()

    stock_df["date"] = pd.to_datetime(stock_df["date"])
    index_df["date"] = pd.to_datetime(index_df["date"])

    breakout_dt = pd.to_datetime(breakout_date)

    # Filter to data up to breakout
    stock_df = stock_df[stock_df["date"] <= breakout_dt].copy()
    index_df = index_df[index_df["date"] <= breakout_dt].copy()

    if len(stock_df) < 60 or len(index_df) < 60:
        return _empty_features()

    # Merge on date
    merged = pd.merge(stock_df, index_df, on="date", suffixes=("_stock", "_index"))
    merged = merged.sort_values("date").reset_index(drop=True)

    if len(merged) < 60:
        return _empty_features()

    # Calculate RS line (stock price / index price, normalized)
    merged["rs_line"] = merged["close_stock"] / merged["close_index"]

    # Normalize RS line to start at 100
    merged["rs_line"] = merged["rs_line"] / merged["rs_line"].iloc[0] * 100

    # RS line slope (4 weeks = 20 trading days)
    rs_4wk = merged["rs_line"].tail(20).values
    if len(rs_4wk) >= 10:
        x = np.arange(len(rs_4wk))
        slope_4wk = np.polyfit(x, rs_4wk, 1)[0]
        # Normalize to % change per week
        rs_line_slope_4wk = (slope_4wk * 5 / rs_4wk[0]) * 100 if rs_4wk[0] > 0 else 0
    else:
        rs_line_slope_4wk = 0.0

    # RS line slope (12 weeks = 60 trading days)
    rs_12wk = merged["rs_line"].tail(60).values
    if len(rs_12wk) >= 30:
        x = np.arange(len(rs_12wk))
        slope_12wk = np.polyfit(x, rs_12wk, 1)[0]
        rs_line_slope_12wk = (slope_12wk * 5 / rs_12wk[0]) * 100 if rs_12wk[0] > 0 else 0
    else:
        rs_line_slope_12wk = 0.0

    # RS acceleration (2nd derivative): is RS accelerating into breakout?
    # Compare recent 2-week slope to prior 2-week slope
    rs_recent = merged["rs_line"].tail(10).values  # Last 2 weeks
    rs_prior = merged["rs_line"].tail(20).head(10).values  # Prior 2 weeks

    if len(rs_recent) >= 5 and len(rs_prior) >= 5:
        x = np.arange(len(rs_recent))
        slope_recent = np.polyfit(x, rs_recent, 1)[0] if len(rs_recent) > 1 else 0
        slope_prior = np.polyfit(x, rs_prior, 1)[0] if len(rs_prior) > 1 else 0
        # Normalize to % per week
        rs_acceleration = ((slope_recent - slope_prior) * 5 / rs_recent[0]) * 100 if rs_recent[0] > 0 else 0
    else:
        rs_acceleration = 0.0

    # RS at new high (within last 52 weeks)
    rs_52wk = merged["rs_line"].tail(252).values
    if len(rs_52wk) > 0:
        rs_new_high = merged["rs_line"].iloc[-1] >= rs_52wk.max() * 0.98
    else:
        rs_new_high = False

    # RS rank percentile (how current RS compares to historical)
    rs_current = merged["rs_line"].iloc[-1]
    rs_historical = merged["rs_line"].values
    rs_rank_percentile = (np.sum(rs_historical <= rs_current) / len(rs_historical)) * 100

    return {
        "rs_line_slope_4wk": float(rs_line_slope_4wk),
        "rs_line_slope_12wk": float(rs_line_slope_12wk),
        "rs_acceleration": float(rs_acceleration),
        "rs_new_high": bool(rs_new_high),
        "rs_rank_percentile": float(rs_rank_percentile),
    }


def _empty_features() -> dict:
    """Return dict of empty/default feature values."""
    return {
        "rs_line_slope_4wk": 0.0,
        "rs_line_slope_12wk": 0.0,
        "rs_acceleration": 0.0,
        "rs_new_high": False,
        "rs_rank_percentile": 50.0,
    }
