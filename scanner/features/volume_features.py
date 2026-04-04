"""Extract volume-based features.

Features: breakout_volume_ratio, volume_trend_in_base,
up_down_volume_ratio, volume_dry_up.
"""

import numpy as np
import pandas as pd

from scanner.config import get


def extract_volume_features(
    df: pd.DataFrame,
    base_start_date: str,
    base_end_date: str,
    breakout_date: str,
) -> dict:
    """Extract volume-based features from price data.

    Args:
        df: DataFrame with columns [date, open, high, low, close, volume].
        base_start_date: Pattern base start date (YYYY-MM-DD).
        base_end_date: Pattern base end date (YYYY-MM-DD).
        breakout_date: Date of breakout (YYYY-MM-DD).

    Returns:
        Dict of volume features.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    vol_avg_period = get("features.volume_avg_period", 50)

    # Get breakout day volume
    breakout_mask = df["date"] == pd.to_datetime(breakout_date)
    if not breakout_mask.any():
        return _empty_features()

    breakout_idx = df[breakout_mask].index[0]
    breakout_volume = df.loc[breakout_idx, "volume"]

    # Calculate 50-day average volume prior to breakout
    prior_start = max(0, breakout_idx - vol_avg_period)
    prior_volumes = df.iloc[prior_start:breakout_idx]["volume"].values

    if len(prior_volumes) < 10:
        return _empty_features()

    avg_volume = np.mean(prior_volumes)
    breakout_volume_ratio = breakout_volume / avg_volume if avg_volume > 0 else 0

    # Volume trend in base (regression slope)
    base_mask = (df["date"] >= base_start_date) & (df["date"] <= base_end_date)
    base_df = df[base_mask]

    if len(base_df) < 5:
        volume_trend = 0.0
    else:
        base_volumes = base_df["volume"].values
        x = np.arange(len(base_volumes))
        # Normalize to percentage change per week
        slope = np.polyfit(x, base_volumes, 1)[0]
        avg_base_vol = np.mean(base_volumes)
        volume_trend = (slope * 5 / avg_base_vol) * 100 if avg_base_vol > 0 else 0

    # Up/down volume ratio
    base_df = base_df.copy()
    base_df["price_change"] = base_df["close"].diff()

    up_volume = base_df[base_df["price_change"] > 0]["volume"].sum()
    down_volume = base_df[base_df["price_change"] < 0]["volume"].sum()

    up_down_ratio = up_volume / down_volume if down_volume > 0 else 2.0

    # Volume dry-up (lowest volume / average volume in base)
    if len(base_df) >= 5:
        min_vol = base_df["volume"].min()
        avg_base_vol = base_df["volume"].mean()
        volume_dry_up = min_vol / avg_base_vol if avg_base_vol > 0 else 1.0
    else:
        volume_dry_up = 1.0

    return {
        "breakout_volume_ratio": float(breakout_volume_ratio),
        "volume_trend_in_base": float(volume_trend),
        "up_down_volume_ratio": float(up_down_ratio),
        "volume_dry_up": float(volume_dry_up),
    }


def _empty_features() -> dict:
    """Return dict of empty/default feature values."""
    return {
        "breakout_volume_ratio": 1.0,
        "volume_trend_in_base": 0.0,
        "up_down_volume_ratio": 1.0,
        "volume_dry_up": 1.0,
    }
