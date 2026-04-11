"""Breakout detection — find all breakout attempts in historical data.

This module finds ALL breakouts using simple criteria (not IBD rules).
The CNN will learn which shapes precede successful breakouts.

Criteria:
1. Price closes above highest close of prior N days
2. Volume > 1.4x 50-day average
3. Stock was in consolidation (range < 35% for 5+ weeks)
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from scanner.config import get, get_price_range
from scanner.db import get_cursor, get_connection

logger = logging.getLogger(__name__)


@dataclass
class Breakout:
    """A detected breakout event."""

    symbol: str
    breakout_date: str
    breakout_price: float
    breakout_volume: int
    volume_ratio: float
    consolidation_start: str
    consolidation_days: int
    consolidation_range_pct: float
    prior_high: float
    prior_high_date: str
    # Legacy outcome (for backward compatibility)
    outcome: Optional[str] = None  # 'success', 'failure', or 'pending'
    outcome_return_pct: Optional[float] = None
    max_gain_pct: Optional[float] = None
    max_loss_pct: Optional[float] = None
    # Multi-label outcomes (for experimentation)
    outcome_asym_20_7: Optional[str] = None   # +20%/-7% (original IBD)
    outcome_asym_15_10: Optional[str] = None  # +15%/-10% (less extreme)
    outcome_sym_10: Optional[str] = None      # +10%/-10% (symmetric)
    return_asym_20_7: Optional[float] = None
    return_asym_15_10: Optional[float] = None
    return_sym_10: Optional[float] = None


def detect_breakouts(
    df: pd.DataFrame,
    symbol: str,
    lookback_days: int = 50,
    min_volume_ratio: float = 1.4,
    min_consolidation_weeks: int = 5,
    max_consolidation_range_pct: float = 35.0,
) -> list[Breakout]:
    """Find all breakout attempts in a stock's price history.

    Args:
        df: DataFrame with columns [date, open, high, low, close, volume].
        symbol: Stock ticker symbol.
        lookback_days: Days to look back for prior high (default 50).
        min_volume_ratio: Minimum breakout volume vs 50-day avg (default 1.4).
        min_consolidation_weeks: Minimum weeks in consolidation (default 5).
        max_consolidation_range_pct: Maximum price range during consolidation (default 35%).

    Returns:
        List of Breakout objects.
    """
    if len(df) < lookback_days + 50:
        return []

    # Load defaults from config if not specified
    if min_volume_ratio == 1.4:
        min_volume_ratio = get("breakout.min_breakout_volume_ratio", 1.4)

    breakouts = []
    df = df.sort_values("date").reset_index(drop=True)

    # Precompute 50-day average volume
    df["vol_avg_50"] = df["volume"].rolling(window=50, min_periods=50).mean()

    # Precompute rolling high over lookback period
    df["prior_high"] = df["close"].shift(1).rolling(window=lookback_days).max()

    min_consolidation_days = min_consolidation_weeks * 5  # Trading days

    for i in range(lookback_days + 50, len(df)):
        row = df.iloc[i]
        close = row["close"]
        volume = row["volume"]
        vol_avg = row["vol_avg_50"]
        prior_high = row["prior_high"]

        if pd.isna(vol_avg) or pd.isna(prior_high):
            continue

        # Check breakout conditions
        # 1. Price closes above prior N-day high
        if close <= prior_high:
            continue

        # 2. Volume > 1.4x 50-day average
        volume_ratio = volume / vol_avg if vol_avg > 0 else 0
        if volume_ratio < min_volume_ratio:
            continue

        # 3. Check for prior consolidation
        consolidation_start_idx = max(0, i - min_consolidation_days)
        consolidation_df = df.iloc[consolidation_start_idx:i]

        if len(consolidation_df) < min_consolidation_days:
            continue

        # Calculate price range during consolidation
        cons_high, cons_low = get_price_range(consolidation_df)
        high_in_cons = cons_high.max()
        low_in_cons = cons_low.min()
        range_pct = ((high_in_cons - low_in_cons) / low_in_cons) * 100 if low_in_cons > 0 else 100

        if range_pct > max_consolidation_range_pct:
            continue

        # Find the prior high date
        prior_high_idx = df.iloc[i - lookback_days : i]["close"].idxmax()
        prior_high_date = df.loc[prior_high_idx, "date"]
        if hasattr(prior_high_date, "strftime"):
            prior_high_date = prior_high_date.strftime("%Y-%m-%d")
        else:
            prior_high_date = str(prior_high_date)[:10]

        # Create breakout record
        breakout_date = row["date"]
        if hasattr(breakout_date, "strftime"):
            breakout_date = breakout_date.strftime("%Y-%m-%d")
        else:
            breakout_date = str(breakout_date)[:10]

        cons_start_date = consolidation_df.iloc[0]["date"]
        if hasattr(cons_start_date, "strftime"):
            cons_start_date = cons_start_date.strftime("%Y-%m-%d")
        else:
            cons_start_date = str(cons_start_date)[:10]

        breakouts.append(
            Breakout(
                symbol=symbol,
                breakout_date=breakout_date,
                breakout_price=float(close),
                breakout_volume=int(volume),
                volume_ratio=float(volume_ratio),
                consolidation_start=cons_start_date,
                consolidation_days=len(consolidation_df),
                consolidation_range_pct=float(range_pct),
                prior_high=float(prior_high),
                prior_high_date=prior_high_date,
            )
        )

    # Remove duplicate breakouts that are too close together (within 5 days)
    breakouts = _dedupe_breakouts(breakouts, min_gap_days=5)

    return breakouts


def _dedupe_breakouts(breakouts: list[Breakout], min_gap_days: int = 5) -> list[Breakout]:
    """Remove breakouts that occur within min_gap_days of each other.

    Keeps the breakout with higher volume ratio.
    """
    if not breakouts:
        return []

    # Sort by date
    breakouts = sorted(breakouts, key=lambda b: b.breakout_date)

    result = [breakouts[0]]
    for b in breakouts[1:]:
        last = result[-1]
        last_date = datetime.strptime(last.breakout_date, "%Y-%m-%d")
        curr_date = datetime.strptime(b.breakout_date, "%Y-%m-%d")

        if (curr_date - last_date).days >= min_gap_days:
            result.append(b)
        elif b.volume_ratio > last.volume_ratio:
            result[-1] = b

    return result


def _compute_outcome_for_thresholds(
    future_df: pd.DataFrame,
    breakout_price: float,
    min_gain_pct: float,
    max_loss_pct: float,
    has_full_window: bool,
) -> tuple[str, float]:
    """Compute outcome for a specific gain/loss threshold pair.

    Args:
        future_df: DataFrame of prices after breakout.
        breakout_price: Price at breakout.
        min_gain_pct: Target gain percentage.
        max_loss_pct: Stop loss percentage.
        has_full_window: Whether full outcome window is available.

    Returns:
        Tuple of (outcome, return_pct).
    """
    hit_target = False
    hit_stop = False

    use_intraday = get("breakout.use_intraday_prices", True)
    for _, row in future_df.iterrows():
        day_high = row["high"] if use_intraday else row["close"]
        day_low = row["low"] if use_intraday else row["close"]
        day_gain = ((day_high - breakout_price) / breakout_price) * 100
        day_loss = ((breakout_price - day_low) / breakout_price) * 100

        if day_gain >= min_gain_pct and not hit_stop:
            hit_target = True
            break
        if day_loss >= max_loss_pct:
            hit_stop = True
            break

    if hit_target:
        return "success", float(min_gain_pct)
    elif hit_stop:
        return "failure", float(-max_loss_pct)
    elif not has_full_window:
        final_price = future_df.iloc[-1]["close"]
        final_return = ((final_price - breakout_price) / breakout_price) * 100
        return "pending", float(final_return)
    else:
        # Didn't hit either threshold — use final price
        final_price = future_df.iloc[-1]["close"]
        final_return = ((final_price - breakout_price) / breakout_price) * 100

        if final_return >= min_gain_pct * 0.5:
            return "success", float(final_return)
        else:
            return "failure", float(final_return)


def label_breakout_outcomes(
    breakouts: list[Breakout],
    df: pd.DataFrame,
    min_gain_pct: float = 20.0,
    max_loss_pct: float = 7.0,
    outcome_window_weeks: int = 12,
    label_mode: str = None,
    symmetric_threshold_pct: float = None,
) -> list[Breakout]:
    """Label breakout outcomes based on subsequent price action.

    Computes ALL label variants in one pass:
    - outcome_asym_20_7: Original IBD +20%/-7%
    - outcome_asym_15_10: Less extreme +15%/-10%
    - outcome_sym_10: Symmetric +10%/-10%

    Also sets legacy 'outcome' field based on config's label_mode.

    Args:
        breakouts: List of Breakout objects to label.
        df: DataFrame with price data extending past breakout dates.
        min_gain_pct: Minimum gain for legacy outcome (default 20%).
        max_loss_pct: Maximum loss for legacy outcome (default 7%).
        outcome_window_weeks: Weeks to track outcome (default 12).
        label_mode: "asymmetric" or "symmetric" for legacy outcome field.
        symmetric_threshold_pct: Threshold for symmetric mode.

    Returns:
        List of Breakout objects with all outcome fields populated.
    """
    # Load defaults from config
    label_mode = label_mode or get("breakout.label_mode", "asymmetric")
    symmetric_threshold_pct = symmetric_threshold_pct or get("breakout.symmetric_threshold_pct", 10.0)
    outcome_window_weeks = get("breakout.outcome_window_weeks", outcome_window_weeks)
    outcome_window_days = outcome_window_weeks * 5  # Trading days

    logger.info(
        "Computing all label variants with %d-week window",
        outcome_window_weeks
    )

    df = df.sort_values("date").reset_index(drop=True)
    df["date_str"] = df["date"].apply(
        lambda d: d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)[:10]
    )

    labeled = []
    for b in breakouts:
        breakout_idx = df[df["date_str"] == b.breakout_date].index
        if len(breakout_idx) == 0:
            continue

        idx = breakout_idx[0]
        end_idx = min(idx + outcome_window_days, len(df))
        has_full_window = end_idx == (idx + outcome_window_days)

        if idx >= len(df) - 1:
            # Not enough future data - mark all as pending
            b.outcome = "pending"
            b.outcome_asym_20_7 = "pending"
            b.outcome_asym_15_10 = "pending"
            b.outcome_sym_10 = "pending"
            labeled.append(b)
            continue

        future_df = df.iloc[idx + 1 : end_idx]
        if future_df.empty:
            b.outcome = "pending"
            b.outcome_asym_20_7 = "pending"
            b.outcome_asym_15_10 = "pending"
            b.outcome_sym_10 = "pending"
            labeled.append(b)
            continue

        # Calculate max gain/loss metrics (shared across all strategies)
        future_high, future_low = get_price_range(future_df)
        max_high = future_high.max()
        min_low = future_low.min()
        b.max_gain_pct = float(((max_high - b.breakout_price) / b.breakout_price) * 100)
        b.max_loss_pct = float(((b.breakout_price - min_low) / b.breakout_price) * 100)

        # Compute all 3 label variants
        # Strategy 1: Original IBD (+20%/-7%)
        outcome_1, return_1 = _compute_outcome_for_thresholds(
            future_df, b.breakout_price, 20.0, 7.0, has_full_window
        )
        b.outcome_asym_20_7 = outcome_1
        b.return_asym_20_7 = return_1

        # Strategy 2: Less extreme (+15%/-10%)
        outcome_2, return_2 = _compute_outcome_for_thresholds(
            future_df, b.breakout_price, 15.0, 10.0, has_full_window
        )
        b.outcome_asym_15_10 = outcome_2
        b.return_asym_15_10 = return_2

        # Strategy 3: Symmetric (+10%/-10%)
        outcome_3, return_3 = _compute_outcome_for_thresholds(
            future_df, b.breakout_price, 10.0, 10.0, has_full_window
        )
        b.outcome_sym_10 = outcome_3
        b.return_sym_10 = return_3

        # Set legacy outcome based on config's label_mode
        if label_mode == "symmetric":
            b.outcome = b.outcome_sym_10
            b.outcome_return_pct = b.return_sym_10
        else:
            b.outcome = b.outcome_asym_20_7
            b.outcome_return_pct = b.return_asym_20_7

        labeled.append(b)

    return labeled


def save_breakouts_to_db(breakouts: list[Breakout], pattern_type: str = "breakout") -> int:
    """Save breakouts to the detected_patterns table.

    Args:
        breakouts: List of Breakout objects.
        pattern_type: Pattern type label (can be updated later by classifier).

    Returns:
        Number of rows inserted.
    """
    count = 0
    with get_cursor() as cur:
        for b in breakouts:
            try:
                cur.execute(
                    """
                    INSERT OR REPLACE INTO detected_patterns
                    (symbol, pattern_type, base_start_date, base_end_date,
                     pivot_date, pivot_price, outcome, outcome_return_pct,
                     outcome_max_gain_pct, outcome_max_loss_pct, auto_label)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        b.symbol,
                        pattern_type,
                        b.consolidation_start,
                        b.breakout_date,
                        b.breakout_date,
                        b.breakout_price,
                        b.outcome,
                        b.outcome_return_pct,
                        b.max_gain_pct,
                        b.max_loss_pct,
                        "auto",
                    ),
                )
                count += 1
            except Exception as e:
                logger.warning("Error saving breakout for %s: %s", b.symbol, e)

    logger.info("Saved %d breakouts to database", count)
    return count


def get_breakouts_from_db(
    symbol: Optional[str] = None,
    outcome: Optional[str] = None,
) -> pd.DataFrame:
    """Load breakouts from database.

    Args:
        symbol: Optional filter by symbol.
        outcome: Optional filter by outcome ('success', 'failure', 'pending').

    Returns:
        DataFrame with breakout data.
    """
    query = "SELECT * FROM detected_patterns WHERE 1=1"
    params = []

    if symbol:
        query += " AND symbol = ?"
        params.append(symbol)
    if outcome:
        query += " AND outcome = ?"
        params.append(outcome)

    query += " ORDER BY pivot_date DESC"

    conn = get_connection()
    try:
        df = pd.read_sql_query(query, conn, params=params)
        return df
    finally:
        conn.close()
