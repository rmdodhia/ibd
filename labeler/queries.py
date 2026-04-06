"""Database query functions for the labeling UI.

Provides paginated pattern retrieval, filtering, and label updates.
"""

import logging
from typing import Optional

import pandas as pd

from scanner.db import get_connection, get_cursor

logger = logging.getLogger(__name__)


def get_patterns_paginated(
    page: int = 1,
    page_size: int = 20,
    symbol: Optional[str] = None,
    pattern_type: Optional[str] = None,
    outcome: Optional[str] = None,
    reviewed: Optional[bool] = None,
) -> tuple[list[dict], int]:
    """Fetch patterns with filters and pagination.

    Args:
        page: Page number (1-indexed).
        page_size: Number of patterns per page.
        symbol: Filter by symbol (exact match).
        pattern_type: Filter by pattern type.
        outcome: Filter by outcome (success/failure/pending).
        reviewed: Filter by reviewed status.

    Returns:
        Tuple of (list of pattern dicts, total count).
    """
    # Build WHERE clause
    conditions = []
    params = []

    if symbol:
        conditions.append("dp.symbol = ?")
        params.append(symbol)
    if pattern_type:
        conditions.append("dp.pattern_type = ?")
        params.append(pattern_type)
    if outcome:
        if outcome == "pending":
            conditions.append("dp.outcome IS NULL")
        else:
            conditions.append("dp.outcome = ?")
            params.append(outcome)
    if reviewed is not None:
        conditions.append("dp.reviewed = ?")
        params.append(1 if reviewed else 0)

    where_clause = " AND ".join(conditions) if conditions else "1=1"

    conn = get_connection()
    try:
        # Get total count
        count_query = f"""
            SELECT COUNT(*) FROM detected_patterns dp
            WHERE {where_clause}
        """
        cur = conn.cursor()
        cur.execute(count_query, params)
        total_count = cur.fetchone()[0]

        # Get paginated results
        offset = (page - 1) * page_size
        query = f"""
            SELECT
                dp.id,
                dp.symbol,
                dp.pattern_type,
                dp.base_start_date,
                dp.base_end_date,
                dp.pivot_date,
                dp.pivot_price,
                dp.outcome,
                dp.outcome_return_pct,
                dp.outcome_max_gain_pct,
                dp.outcome_max_loss_pct,
                dp.auto_label,
                dp.human_label,
                dp.reviewed,
                dp.created_at
            FROM detected_patterns dp
            WHERE {where_clause}
            ORDER BY dp.pivot_date DESC, dp.symbol ASC
            LIMIT ? OFFSET ?
        """
        params.extend([page_size, offset])

        cur.execute(query, params)
        rows = cur.fetchall()

        patterns = []
        for row in rows:
            patterns.append({
                "id": row[0],
                "symbol": row[1],
                "pattern_type": row[2],
                "base_start_date": row[3],
                "base_end_date": row[4],
                "pivot_date": row[5],
                "pivot_price": row[6],
                "outcome": row[7],
                "outcome_return_pct": row[8],
                "outcome_max_gain_pct": row[9],
                "outcome_max_loss_pct": row[10],
                "auto_label": row[11],
                "human_label": row[12],
                "reviewed": bool(row[13]),
                "created_at": row[14],
            })

        return patterns, total_count

    finally:
        conn.close()


def get_pattern_with_features(pattern_id: int) -> Optional[dict]:
    """Fetch a single pattern with all its features.

    Args:
        pattern_id: Pattern ID.

    Returns:
        Dict with pattern data and features, or None if not found.
    """
    conn = get_connection()
    try:
        # Check if multi-label columns exist
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(detected_patterns)")
        columns = {row[1] for row in cur.fetchall()}
        has_multi_labels = "outcome_asym_20_7" in columns

        query = """
            SELECT
                dp.id,
                dp.symbol,
                dp.pattern_type,
                dp.base_start_date,
                dp.base_end_date,
                dp.pivot_date,
                dp.pivot_price,
                dp.outcome,
                dp.outcome_return_pct,
                dp.outcome_max_gain_pct,
                dp.outcome_max_loss_pct,
                dp.auto_label,
                dp.human_label,
                dp.reviewed,
                dp.created_at,
                """ + ("""
                dp.outcome_asym_20_7,
                dp.outcome_asym_15_10,
                dp.outcome_sym_10,
                dp.return_asym_20_7,
                dp.return_asym_15_10,
                dp.return_sym_10,
                """ if has_multi_labels else """
                NULL as outcome_asym_20_7,
                NULL as outcome_asym_15_10,
                NULL as outcome_sym_10,
                NULL as return_asym_20_7,
                NULL as return_asym_15_10,
                NULL as return_sym_10,
                """) + """
                pf.base_depth_pct,
                pf.base_duration_weeks,
                pf.base_symmetry,
                pf.handle_depth_pct,
                pf.tightness_score,
                pf.breakout_volume_ratio,
                pf.volume_trend_in_base,
                pf.up_down_volume_ratio,
                pf.rs_line_slope_4wk,
                pf.rs_line_slope_12wk,
                pf.rs_new_high,
                pf.rs_rank_percentile,
                pf.eps_latest_yoy_growth,
                pf.eps_acceleration,
                pf.revenue_latest_yoy_growth,
                pf.institutional_pct,
                pf.market_cap_log,
                pf.sp500_above_200dma,
                pf.sp500_trend_4wk,
                pf.price_vs_50dma,
                pf.price_vs_200dma
            FROM detected_patterns dp
            LEFT JOIN pattern_features pf ON dp.id = pf.pattern_id
            WHERE dp.id = ?
        """
        cur = conn.cursor()
        cur.execute(query, (pattern_id,))
        row = cur.fetchone()

        if not row:
            return None

        # Row indices (accounting for 6 new multi-label columns after index 14)
        return {
            "id": row[0],
            "symbol": row[1],
            "pattern_type": row[2],
            "base_start_date": row[3],
            "base_end_date": row[4],
            "pivot_date": row[5],
            "pivot_price": row[6],
            "outcome": row[7],
            "outcome_return_pct": row[8],
            "outcome_max_gain_pct": row[9],
            "outcome_max_loss_pct": row[10],
            "auto_label": row[11],
            "human_label": row[12],
            "reviewed": bool(row[13]),
            "created_at": row[14],
            # Multi-label outcomes (indices 15-20)
            "outcome_asym_20_7": row[15],
            "outcome_asym_15_10": row[16],
            "outcome_sym_10": row[17],
            "return_asym_20_7": row[18],
            "return_asym_15_10": row[19],
            "return_sym_10": row[20],
            "features": {
                "base_depth_pct": row[21],
                "base_duration_weeks": row[22],
                "base_symmetry": row[23],
                "handle_depth_pct": row[24],
                "tightness_score": row[25],
                "breakout_volume_ratio": row[26],
                "volume_trend_in_base": row[27],
                "up_down_volume_ratio": row[28],
                "rs_line_slope_4wk": row[29],
                "rs_line_slope_12wk": row[30],
                "rs_new_high": bool(row[31]) if row[31] is not None else None,
                "rs_rank_percentile": row[32],
                "eps_latest_yoy_growth": row[33],
                "eps_acceleration": row[34],
                "revenue_latest_yoy_growth": row[35],
                "institutional_pct": row[36],
                "market_cap_log": row[37],
                "sp500_above_200dma": bool(row[38]) if row[38] is not None else None,
                "sp500_trend_4wk": row[39],
                "price_vs_50dma": row[40],
                "price_vs_200dma": row[41],
            },
        }

    finally:
        conn.close()


def update_pattern_label(
    pattern_id: int,
    human_label: str,
    reviewed: bool = True,
) -> bool:
    """Update human_label and reviewed status for a pattern.

    Args:
        pattern_id: Pattern ID to update.
        human_label: New human label (success/failure/ambiguous).
        reviewed: Mark as reviewed.

    Returns:
        True if update succeeded, False otherwise.
    """
    try:
        with get_cursor() as cur:
            cur.execute(
                """
                UPDATE detected_patterns
                SET human_label = ?, reviewed = ?
                WHERE id = ?
                """,
                (human_label, 1 if reviewed else 0, pattern_id),
            )
            return cur.rowcount > 0
    except Exception as e:
        logger.error("Failed to update pattern %d: %s", pattern_id, e)
        return False


def get_progress_stats() -> dict:
    """Get labeling progress statistics.

    Returns:
        Dict with total, reviewed, unreviewed counts and breakdown by outcome.
    """
    conn = get_connection()
    try:
        cur = conn.cursor()

        # Total count
        cur.execute("SELECT COUNT(*) FROM detected_patterns")
        total = cur.fetchone()[0]

        # Reviewed count
        cur.execute("SELECT COUNT(*) FROM detected_patterns WHERE reviewed = 1")
        reviewed = cur.fetchone()[0]

        # Breakdown by outcome
        cur.execute("""
            SELECT outcome, COUNT(*) as cnt
            FROM detected_patterns
            GROUP BY outcome
        """)
        outcome_counts = {row[0]: row[1] for row in cur.fetchall()}

        # Breakdown by human_label (for reviewed patterns)
        cur.execute("""
            SELECT human_label, COUNT(*) as cnt
            FROM detected_patterns
            WHERE reviewed = 1
            GROUP BY human_label
        """)
        label_counts = {row[0]: row[1] for row in cur.fetchall()}

        # Agreement rate (where human_label matches auto_label)
        cur.execute("""
            SELECT COUNT(*) FROM detected_patterns
            WHERE reviewed = 1 AND human_label = auto_label
        """)
        agreed = cur.fetchone()[0]

        return {
            "total": total,
            "reviewed": reviewed,
            "unreviewed": total - reviewed,
            "pct_complete": (reviewed / total * 100) if total > 0 else 0,
            "outcome_counts": outcome_counts,
            "label_counts": label_counts,
            "agreement_rate": (agreed / reviewed * 100) if reviewed > 0 else 0,
        }

    finally:
        conn.close()


def get_disagreement_stats() -> dict:
    """Get disagreement statistics between auto and human labels.

    Returns:
        Dict with disagreement breakdown by strategy and pattern type.
    """
    conn = get_connection()
    try:
        cur = conn.cursor()

        # Check if we have any reviewed patterns
        cur.execute("""
            SELECT COUNT(*) FROM detected_patterns
            WHERE reviewed = 1 AND human_label IN ('success', 'failure')
        """)
        n_reviewed = cur.fetchone()[0]

        if n_reviewed < 5:
            return {"n_reviewed": n_reviewed, "insufficient_data": True}

        result = {
            "n_reviewed": n_reviewed,
            "insufficient_data": False,
            "by_strategy": {},
            "by_pattern_type": {},
            "disagreement_types": {},
        }

        # Agreement by label strategy
        strategies = [
            ("outcome_asym_20_7", "+20%/-7%"),
            ("outcome_asym_15_10", "+15%/-10%"),
            ("outcome_sym_10", "+/-10%"),
        ]

        # Check which columns exist
        cur.execute("PRAGMA table_info(detected_patterns)")
        columns = {row[1] for row in cur.fetchall()}

        for col, name in strategies:
            if col not in columns:
                continue

            cur.execute(f"""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN human_label = {col} THEN 1 ELSE 0 END) as agreed
                FROM detected_patterns
                WHERE reviewed = 1
                  AND human_label IN ('success', 'failure')
                  AND {col} IN ('success', 'failure')
            """)
            row = cur.fetchone()
            total, agreed = row[0] or 0, row[1] or 0

            if total > 0:
                result["by_strategy"][col] = {
                    "name": name,
                    "total": total,
                    "agreed": agreed,
                    "rate": agreed / total * 100,
                }

        # Agreement by pattern type
        cur.execute("""
            SELECT
                pattern_type,
                COUNT(*) as total,
                SUM(CASE WHEN human_label = outcome THEN 1 ELSE 0 END) as agreed
            FROM detected_patterns
            WHERE reviewed = 1
              AND human_label IN ('success', 'failure')
              AND outcome IN ('success', 'failure')
            GROUP BY pattern_type
            HAVING total >= 3
        """)
        for row in cur.fetchall():
            ptype, total, agreed = row
            result["by_pattern_type"][ptype] = {
                "total": total,
                "agreed": agreed,
                "rate": agreed / total * 100 if total > 0 else 0,
            }

        # Disagreement breakdown
        cur.execute("""
            SELECT
                SUM(CASE WHEN human_label = 'success' AND outcome = 'failure' THEN 1 ELSE 0 END),
                SUM(CASE WHEN human_label = 'failure' AND outcome = 'success' THEN 1 ELSE 0 END)
            FROM detected_patterns
            WHERE reviewed = 1
              AND human_label IN ('success', 'failure')
              AND outcome IN ('success', 'failure')
        """)
        row = cur.fetchone()
        result["disagreement_types"] = {
            "human_success_auto_failure": row[0] or 0,
            "human_failure_auto_success": row[1] or 0,
        }

        return result

    finally:
        conn.close()


def get_distinct_symbols() -> list[str]:
    """Get all distinct symbols with detected patterns.

    Returns:
        Sorted list of symbol strings.
    """
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT DISTINCT symbol FROM detected_patterns
            ORDER BY symbol ASC
        """)
        return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()


def get_distinct_pattern_types() -> list[str]:
    """Get all distinct pattern types.

    Returns:
        List of pattern type strings.
    """
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT DISTINCT pattern_type FROM detected_patterns
            ORDER BY pattern_type ASC
        """)
        return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()


def get_price_data_for_chart(
    symbol: str,
    base_start_date: str,
    pivot_date: str,
    days_before: int = 60,
    days_after: int = 60,
) -> pd.DataFrame:
    """Get price data for charting a pattern.

    Fetches data from days_before the base_start_date to days_after the pivot_date.

    Args:
        symbol: Stock symbol.
        base_start_date: Pattern base start date.
        pivot_date: Pattern pivot date.
        days_before: Days of context before base starts.
        days_after: Days of context after pivot.

    Returns:
        DataFrame with OHLCV data.
    """
    conn = get_connection()
    try:
        query = """
            SELECT date, open, high, low, close, volume
            FROM daily_prices
            WHERE symbol = ?
              AND date >= date(?, '-' || ? || ' days')
              AND date <= date(?, '+' || ? || ' days')
            ORDER BY date ASC
        """
        df = pd.read_sql_query(
            query,
            conn,
            params=(symbol, base_start_date, days_before, pivot_date, days_after),
        )
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df
    finally:
        conn.close()
