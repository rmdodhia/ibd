"""Extract market context features.

Features: sp500_above_200dma, sp500_trend_4wk, price_vs_50dma, price_vs_200dma.
"""

import numpy as np
import pandas as pd

from scanner.config import get


def extract_market_features(
    stock_df: pd.DataFrame,
    index_df: pd.DataFrame,
    breakout_date: str,
) -> dict:
    """Extract market context features.

    Args:
        stock_df: Stock price DataFrame with [date, close].
        index_df: Index price DataFrame (S&P 500) with [date, close].
        breakout_date: Date of breakout (YYYY-MM-DD).

    Returns:
        Dict of market context features.
    """
    stock_df = stock_df.copy()
    index_df = index_df.copy()

    stock_df["date"] = pd.to_datetime(stock_df["date"])
    index_df["date"] = pd.to_datetime(index_df["date"])

    breakout_dt = pd.to_datetime(breakout_date)

    # Filter to data up to breakout
    stock_df = stock_df[stock_df["date"] <= breakout_dt].sort_values("date").reset_index(drop=True)
    index_df = index_df[index_df["date"] <= breakout_dt].sort_values("date").reset_index(drop=True)

    moving_averages = get("features.moving_averages", [50, 200])

    # S&P 500 features
    if len(index_df) >= 200:
        # S&P 500 above 200 DMA
        index_200dma = index_df["close"].tail(200).mean()
        index_current = index_df["close"].iloc[-1]
        sp500_above_200dma = index_current > index_200dma

        # S&P 500 trend (4 weeks)
        index_4wk_ago = index_df["close"].iloc[-20] if len(index_df) >= 20 else index_df["close"].iloc[0]
        sp500_trend_4wk = ((index_current - index_4wk_ago) / index_4wk_ago) * 100
    else:
        sp500_above_200dma = True
        sp500_trend_4wk = 0.0

    # Stock price vs moving averages
    if len(stock_df) >= 200:
        stock_current = stock_df["close"].iloc[-1]

        # Price vs 50 DMA
        ma_50 = stock_df["close"].tail(50).mean()
        price_vs_50dma = ((stock_current - ma_50) / ma_50) * 100 if ma_50 > 0 else 0

        # Price vs 200 DMA
        ma_200 = stock_df["close"].tail(200).mean()
        price_vs_200dma = ((stock_current - ma_200) / ma_200) * 100 if ma_200 > 0 else 0
    elif len(stock_df) >= 50:
        stock_current = stock_df["close"].iloc[-1]
        ma_50 = stock_df["close"].tail(50).mean()
        price_vs_50dma = ((stock_current - ma_50) / ma_50) * 100 if ma_50 > 0 else 0
        price_vs_200dma = 0.0
    else:
        price_vs_50dma = 0.0
        price_vs_200dma = 0.0

    return {
        "sp500_above_200dma": bool(sp500_above_200dma),
        "sp500_trend_4wk": float(sp500_trend_4wk),
        "price_vs_50dma": float(price_vs_50dma),
        "price_vs_200dma": float(price_vs_200dma),
    }


def _empty_features() -> dict:
    """Return dict of empty/default feature values."""
    return {
        "sp500_above_200dma": True,
        "sp500_trend_4wk": 0.0,
        "price_vs_50dma": 0.0,
        "price_vs_200dma": 0.0,
    }
