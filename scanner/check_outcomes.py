"""Check and update outcomes for pending predictions.

Pulls latest price data and determines if predictions
have resolved as success or failure.

Usage:
    python -m scanner.check_outcomes
"""

import logging
from datetime import datetime, timedelta

import pandas as pd

from scanner.config import get, get_price_range
from scanner.db import get_cursor, get_connection, init_db
from scanner.data_pipeline import get_price_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def check_outcomes() -> dict:
    """Check and update outcomes for all pending predictions.

    Returns:
        Dict with update statistics.
    """
    init_db()

    min_gain_pct = get("breakout.min_gain_pct", 20)
    max_loss_pct = get("breakout.max_loss_pct", 7)
    outcome_window_weeks = get("breakout.outcome_window_weeks", 8)
    outcome_window_days = outcome_window_weeks * 7

    # Get pending predictions
    conn = get_connection()
    try:
        pending = pd.read_sql_query(
            """
            SELECT id, symbol, prediction_date, confidence_score
            FROM predictions
            WHERE actual_outcome IS NULL
            ORDER BY prediction_date
            """,
            conn,
        )
    finally:
        conn.close()

    if pending.empty:
        logger.info("No pending predictions to check")
        return {"checked": 0, "resolved": 0}

    logger.info("Checking %d pending predictions...", len(pending))

    stats = {
        "checked": 0,
        "resolved": 0,
        "success": 0,
        "failure": 0,
        "still_pending": 0,
    }

    for _, row in pending.iterrows():
        result = _check_single_prediction(
            row["id"],
            row["symbol"],
            row["prediction_date"],
            row["confidence_score"],
            min_gain_pct,
            max_loss_pct,
            outcome_window_days,
        )
        stats["checked"] += 1
        if result == "success":
            stats["resolved"] += 1
            stats["success"] += 1
        elif result == "failure":
            stats["resolved"] += 1
            stats["failure"] += 1
        else:
            stats["still_pending"] += 1

    logger.info(
        "Checked %d predictions: %d resolved (%d success, %d failure), %d still pending",
        stats["checked"],
        stats["resolved"],
        stats["success"],
        stats["failure"],
        stats["still_pending"],
    )

    return stats


def _check_single_prediction(
    pred_id: int,
    symbol: str,
    prediction_date: str,
    confidence: float,
    min_gain_pct: float,
    max_loss_pct: float,
    outcome_window_days: int,
) -> str:
    """Check outcome for a single prediction.

    Returns:
        'success', 'failure', or 'pending'.
    """
    # Get price data
    stock_df = get_price_data(symbol)
    if stock_df.empty:
        return "pending"

    stock_df["date"] = pd.to_datetime(stock_df["date"])
    pred_dt = pd.to_datetime(prediction_date)

    # Get prediction day price (entry price)
    pred_day = stock_df[stock_df["date"] == pred_dt]
    if pred_day.empty:
        # Try next available day
        future = stock_df[stock_df["date"] > pred_dt].head(1)
        if future.empty:
            return "pending"
        entry_price = future["close"].iloc[0]
        entry_date = future["date"].iloc[0]
    else:
        entry_price = pred_day["close"].iloc[0]
        entry_date = pred_dt

    # Get price data after entry
    future_df = stock_df[stock_df["date"] > entry_date].head(outcome_window_days)

    if future_df.empty:
        return "pending"

    # Check if enough time has passed
    max_date = future_df["date"].max()
    days_elapsed = (max_date - entry_date).days

    # Calculate max gain and max loss
    future_high, future_low = get_price_range(future_df)
    max_high = future_high.max()
    min_low = future_low.min()

    max_gain = ((max_high - entry_price) / entry_price) * 100
    max_loss = ((entry_price - min_low) / entry_price) * 100

    # Determine outcome
    outcome = None
    actual_return = None

    # Check day by day for sequence
    hit_target = False
    hit_stop = False

    use_intraday = get("breakout.use_intraday_prices", True)
    for _, day in future_df.iterrows():
        day_high = day["high"] if use_intraday else day["close"]
        day_low = day["low"] if use_intraday else day["close"]
        day_gain = ((day_high - entry_price) / entry_price) * 100
        day_loss = ((entry_price - day_low) / entry_price) * 100

        if day_gain >= min_gain_pct and not hit_stop:
            hit_target = True
            outcome = "success"
            actual_return = min_gain_pct
            break
        if day_loss >= max_loss_pct:
            hit_stop = True
            outcome = "failure"
            actual_return = -max_loss_pct
            break

    # If neither threshold hit, check if enough time passed
    if outcome is None:
        if days_elapsed >= outcome_window_days:
            # Use final return
            final_price = future_df["close"].iloc[-1]
            actual_return = ((final_price - entry_price) / entry_price) * 100

            if actual_return >= min_gain_pct / 2:  # Partial success
                outcome = "success"
            elif actual_return <= -max_loss_pct / 2:
                outcome = "failure"
            else:
                outcome = "neutral"
        else:
            return "pending"

    # Update database
    with get_cursor() as cur:
        cur.execute(
            """
            UPDATE predictions
            SET actual_outcome = ?,
                actual_return_pct = ?,
                resolved_date = ?
            WHERE id = ?
            """,
            (
                outcome,
                actual_return,
                datetime.now().strftime("%Y-%m-%d"),
                pred_id,
            ),
        )

    return outcome


def main():
    stats = check_outcomes()

    print("\n" + "=" * 50)
    print("OUTCOME CHECK RESULTS")
    print("=" * 50)
    print(f"Predictions checked: {stats['checked']}")
    print(f"Newly resolved: {stats['resolved']}")
    print(f"  - Success: {stats['success']}")
    print(f"  - Failure: {stats['failure']}")
    print(f"Still pending: {stats['still_pending']}")
    print("=" * 50)


if __name__ == "__main__":
    main()
