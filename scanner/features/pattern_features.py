"""Extract geometry features from detected patterns.

Features: base_depth_pct, base_duration_weeks, base_symmetry,
handle_depth_pct, tightness_score, support_touches, resistance_touches.
"""

import numpy as np
import pandas as pd


def extract_pattern_features(
    df: pd.DataFrame,
    base_start_date: str,
    base_end_date: str,
    pattern_metadata: dict,
) -> dict:
    """Extract geometric features from a pattern's price data.

    Args:
        df: DataFrame with columns [date, open, high, low, close, volume].
        base_start_date: Pattern base start date (YYYY-MM-DD).
        base_end_date: Pattern base end date (YYYY-MM-DD).
        pattern_metadata: Additional metadata from pattern detector.

    Returns:
        Dict of pattern geometry features.
    """
    # Filter to pattern period
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    mask = (df["date"] >= base_start_date) & (df["date"] <= base_end_date)
    pattern_df = df[mask].copy()

    if len(pattern_df) < 5:
        return _empty_features()

    highs = pattern_df["high"].values
    lows = pattern_df["low"].values
    closes = pattern_df["close"].values

    # Base depth
    high_price = highs.max()
    low_price = lows.min()
    base_depth_pct = ((high_price - low_price) / high_price) * 100 if high_price > 0 else 0

    # Base duration in weeks
    duration_days = len(pattern_df)
    base_duration_weeks = duration_days / 5.0

    # Symmetry: time to trough vs time from trough
    trough_idx = np.argmin(lows)
    left_duration = trough_idx
    right_duration = len(lows) - trough_idx - 1

    if left_duration > 0 and right_duration > 0:
        base_symmetry = min(left_duration, right_duration) / max(left_duration, right_duration)
    else:
        base_symmetry = 0.5

    # Handle depth (from metadata if available)
    handle_depth_pct = pattern_metadata.get("handle_depth_pct", 0.0)

    # Tightness: coefficient of variation of weekly closes
    weekly_closes = closes[::5] if len(closes) >= 5 else closes
    if len(weekly_closes) > 1 and np.mean(weekly_closes) > 0:
        cv = np.std(weekly_closes) / np.mean(weekly_closes)
        tightness_score = max(0, min(1, 1 - cv / 0.1))
    else:
        tightness_score = 0.5

    # Pre-breakout tightness: price range contraction in final 2-3 weeks
    # IBD emphasizes this as a key signal - consolidation should tighten before breakout
    final_weeks_days = min(15, len(pattern_df))  # Last 3 weeks
    final_df = pattern_df.tail(final_weeks_days)

    if len(final_df) >= 5:
        # Calculate daily ranges in final period
        final_ranges = final_df["high"].values - final_df["low"].values
        final_range_avg = np.mean(final_ranges) if len(final_ranges) > 0 else 0

        # Calculate daily ranges in earlier period
        earlier_df = pattern_df.head(len(pattern_df) - final_weeks_days)
        if len(earlier_df) >= 5:
            earlier_ranges = earlier_df["high"].values - earlier_df["low"].values
            earlier_range_avg = np.mean(earlier_ranges) if len(earlier_ranges) > 0 else 0

            # Pre-breakout tightness ratio: lower = more contraction (better)
            if earlier_range_avg > 0:
                pre_breakout_tightness = final_range_avg / earlier_range_avg
            else:
                pre_breakout_tightness = 1.0
        else:
            pre_breakout_tightness = 1.0

        # Also calculate as % of price (ATR-like)
        avg_price = final_df["close"].mean()
        if avg_price > 0:
            pre_breakout_range_pct = (final_range_avg / avg_price) * 100
        else:
            pre_breakout_range_pct = 0.0
    else:
        pre_breakout_tightness = 1.0
        pre_breakout_range_pct = 0.0

    # Support/resistance touches
    support_level = low_price * 1.02  # Within 2% of low
    resistance_level = high_price * 0.98  # Within 2% of high

    support_touches = np.sum(lows <= support_level)
    resistance_touches = np.sum(highs >= resistance_level)

    return {
        "base_depth_pct": float(base_depth_pct),
        "base_duration_weeks": float(base_duration_weeks),
        "base_symmetry": float(base_symmetry),
        "handle_depth_pct": float(handle_depth_pct),
        "tightness_score": float(tightness_score),
        "pre_breakout_tightness": float(pre_breakout_tightness),
        "pre_breakout_range_pct": float(pre_breakout_range_pct),
        "support_touches": int(support_touches),
        "resistance_touches": int(resistance_touches),
    }


def _empty_features() -> dict:
    """Return dict of empty/default feature values."""
    return {
        "base_depth_pct": 0.0,
        "base_duration_weeks": 0.0,
        "base_symmetry": 0.5,
        "handle_depth_pct": 0.0,
        "tightness_score": 0.5,
        "pre_breakout_tightness": 1.0,
        "pre_breakout_range_pct": 0.0,
        "support_touches": 0,
        "resistance_touches": 0,
    }
