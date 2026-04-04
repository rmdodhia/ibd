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

from scanner.config import get
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
    outcome: Optional[str] = None  # 'success', 'failure', 'pending'
    outcome_return_pct: Optional[float] = None
    max_gain_pct: Optional[float] = None
    max_loss_pct: Optional[float] = None


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
        high_in_cons = consolidation_df["high"].max()
        low_in_cons = consolidation_df["low"].min()
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


def label_breakout_outcomes(
    breakouts: list[Breakout],
    df: pd.DataFrame,
    min_gain_pct: float = 20.0,
    max_loss_pct: float = 7.0,
    outcome_window_weeks: int = 8,
) -> list[Breakout]:
    """Label breakout outcomes based on subsequent price action.

    Args:
        breakouts: List of Breakout objects to label.
        df: DataFrame with price data extending past breakout dates.
        min_gain_pct: Minimum gain to count as success (default 20%).
        max_loss_pct: Maximum loss before counted as failure (default 7%).
        outcome_window_weeks: Weeks to track outcome (default 8).

    Returns:
        List of Breakout objects with outcome fields populated.
    """
    # Load defaults from config
    min_gain_pct = get("breakout.min_gain_pct", min_gain_pct)
    max_loss_pct = get("breakout.max_loss_pct", max_loss_pct)
    outcome_window_weeks = get("breakout.outcome_window_weeks", outcome_window_weeks)

    outcome_window_days = outcome_window_weeks * 5  # Trading days

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

        if idx >= len(df) - 1:
            # Not enough future data
            b.outcome = "pending"
            labeled.append(b)
            continue

        future_df = df.iloc[idx + 1 : end_idx]
        if future_df.empty:
            b.outcome = "pending"
            labeled.append(b)
            continue

        # Calculate metrics
        max_high = future_df["high"].max()
        min_low = future_df["low"].min()

        max_gain = ((max_high - b.breakout_price) / b.breakout_price) * 100
        max_loss = ((b.breakout_price - min_low) / b.breakout_price) * 100

        b.max_gain_pct = float(max_gain)
        b.max_loss_pct = float(max_loss)

        # Determine outcome
        # Success: hit target gain before hitting stop loss
        # Failure: hit stop loss first
        hit_target = False
        hit_stop = False

        for _, row in future_df.iterrows():
            day_gain = ((row["high"] - b.breakout_price) / b.breakout_price) * 100
            day_loss = ((b.breakout_price - row["low"]) / b.breakout_price) * 100

            if day_gain >= min_gain_pct and not hit_stop:
                hit_target = True
                break
            if day_loss >= max_loss_pct:
                hit_stop = True
                break

        if hit_target:
            b.outcome = "success"
            b.outcome_return_pct = float(min_gain_pct)
        elif hit_stop:
            b.outcome = "failure"
            b.outcome_return_pct = float(-max_loss_pct)
        else:
            # Didn't hit either threshold — use final price
            final_price = future_df.iloc[-1]["close"]
            final_return = ((final_price - b.breakout_price) / b.breakout_price) * 100
            b.outcome_return_pct = float(final_return)

            if final_return >= min_gain_pct * 0.5:  # 10%+ is partial success
                b.outcome = "success"
            elif final_return <= -max_loss_pct * 0.5:  # -3.5%+ is failure
                b.outcome = "failure"
            else:
                b.outcome = "neutral"

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
