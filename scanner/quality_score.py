"""Integrated quality scoring for breakout patterns.

Combines technical, fundamental, and market factors into a single
quality score that predicts breakout success probability.

Based on CAN SLIM methodology:
- C: Current quarterly EPS (up 25%+)
- A: Annual EPS growth (up 25%+)
- N: New price highs, RS line at new high
- S: Supply/demand (volume patterns)
- L: Leader (RS rank in top 20%)
- I: Institutional sponsorship
- M: Market direction (confirmed uptrend)
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from scanner.config import get

logger = logging.getLogger(__name__)


@dataclass
class QualityScore:
    """Quality assessment for a breakout pattern."""

    total_score: float  # 0-1, overall quality
    technical_score: float  # 0-1, pattern/volume/RS quality
    fundamental_score: float  # 0-1, CAN SLIM fundamentals
    market_score: float  # 0-1, market conditions

    # Individual criteria pass/fail
    has_prior_uptrend: bool
    has_volume_confirmation: bool
    has_tight_action: bool
    rs_at_high: bool
    eps_growing: bool
    revenue_growing: bool
    institutional_support: bool
    market_uptrend: bool

    # Detailed scores
    prior_uptrend_pct: float
    volume_ratio: float
    tightness: float
    rs_rank: float
    eps_growth: float
    revenue_growth: float

    def passes_minimum(self, min_score: float = 0.6) -> bool:
        """Check if pattern meets minimum quality threshold."""
        return self.total_score >= min_score

    def passes_canslim(self, min_criteria: int = 4) -> bool:
        """Check if enough CAN SLIM criteria pass."""
        criteria_passed = sum([
            self.eps_growing,
            self.has_prior_uptrend,
            self.rs_at_high,
            self.has_volume_confirmation,
            self.institutional_support,
            self.market_uptrend,
        ])
        return criteria_passed >= min_criteria


def compute_quality_score(
    stock_df: pd.DataFrame,
    index_df: pd.DataFrame,
    features: dict,
    base_start_date: str,
    base_end_date: str,
) -> QualityScore:
    """Compute comprehensive quality score for a pattern.

    Args:
        stock_df: Stock price DataFrame.
        index_df: Index (S&P 500) price DataFrame.
        features: Dict of extracted features from feature extractors.
        base_start_date: Pattern start date.
        base_end_date: Pattern end date (breakout date).

    Returns:
        QualityScore object with detailed assessment.
    """
    # === TECHNICAL SCORE (40% weight) ===

    # 1. Prior uptrend check (30%+ gain in 6-12 months before pattern)
    prior_uptrend_pct = _check_prior_uptrend(stock_df, base_start_date)
    has_prior_uptrend = prior_uptrend_pct >= 30.0

    # 2. Volume confirmation
    volume_ratio = features.get("breakout_volume_ratio", 1.0)
    up_down_ratio = features.get("up_down_volume_ratio", 1.0)
    volume_trend = features.get("volume_trend_in_base", 0.0)

    # Good: breakout volume 50%+ above average, declining volume in base
    has_volume_confirmation = (
        volume_ratio >= 1.5 and
        volume_trend <= 0 and  # Declining or stable
        up_down_ratio >= 0.8
    )

    # 3. Tight price action near pivot
    tightness = features.get("tightness_score", 0.5)
    has_tight_action = tightness >= 0.6

    # 4. Relative strength
    rs_rank = features.get("rs_rank_percentile", 50.0)
    rs_new_high = features.get("rs_new_high", False)
    rs_slope_4wk = features.get("rs_line_slope_4wk", 0.0)

    rs_at_high = rs_new_high or (rs_rank >= 80 and rs_slope_4wk > 0)

    # Calculate technical score
    tech_components = []

    # Prior uptrend (0-25 points)
    if prior_uptrend_pct >= 100:
        tech_components.append(25)
    elif prior_uptrend_pct >= 50:
        tech_components.append(20)
    elif prior_uptrend_pct >= 30:
        tech_components.append(15)
    else:
        tech_components.append(max(0, prior_uptrend_pct / 2))

    # Volume confirmation (0-25 points)
    vol_score = 0
    if volume_ratio >= 2.0:
        vol_score += 15
    elif volume_ratio >= 1.5:
        vol_score += 10
    elif volume_ratio >= 1.2:
        vol_score += 5
    if volume_trend <= -5:  # Declining volume in base
        vol_score += 5
    if up_down_ratio >= 1.2:
        vol_score += 5
    tech_components.append(min(25, vol_score))

    # Tightness (0-25 points)
    tech_components.append(tightness * 25)

    # RS strength (0-25 points)
    rs_score = 0
    if rs_new_high:
        rs_score += 15
    if rs_rank >= 90:
        rs_score += 10
    elif rs_rank >= 80:
        rs_score += 7
    elif rs_rank >= 70:
        rs_score += 4
    tech_components.append(min(25, rs_score))

    technical_score = sum(tech_components) / 100.0

    # === FUNDAMENTAL SCORE (40% weight) ===

    eps_growth = features.get("eps_latest_yoy_growth", 0.0)
    eps_accel = features.get("eps_acceleration", 0.0)
    revenue_growth = features.get("revenue_latest_yoy_growth", 0.0)
    institutional_pct = features.get("institutional_pct", 0.5)

    # Check if we have fundamental data (all zeros means missing)
    has_fundamental_data = (eps_growth != 0.0 or revenue_growth != 0.0)

    eps_growing = eps_growth >= 25.0
    revenue_growing = revenue_growth >= 20.0
    institutional_support = institutional_pct >= 0.4

    fund_components = []

    if has_fundamental_data:
        # EPS growth (0-35 points)
        if eps_growth >= 50:
            fund_components.append(35)
        elif eps_growth >= 25:
            fund_components.append(25)
        elif eps_growth >= 15:
            fund_components.append(15)
        elif eps_growth > 0:
            fund_components.append(8)
        else:
            fund_components.append(0)

        # EPS acceleration bonus (0-15 points)
        if eps_accel >= 10:
            fund_components.append(15)
        elif eps_accel >= 5:
            fund_components.append(10)
        elif eps_accel > 0:
            fund_components.append(5)
        else:
            fund_components.append(0)

        # Revenue growth (0-30 points)
        if revenue_growth >= 25:
            fund_components.append(30)
        elif revenue_growth >= 15:
            fund_components.append(20)
        elif revenue_growth > 0:
            fund_components.append(10)
        else:
            fund_components.append(0)
    else:
        # No fundamental data - use neutral score (don't penalize)
        fund_components.append(17)  # Neutral EPS
        fund_components.append(7)   # Neutral acceleration
        fund_components.append(15)  # Neutral revenue
        eps_growing = True  # Assume OK when no data
        revenue_growing = True

    # Institutional ownership (0-20 points)
    if institutional_pct >= 0.7:
        fund_components.append(20)
    elif institutional_pct >= 0.5:
        fund_components.append(15)
    elif institutional_pct >= 0.3:
        fund_components.append(10)
    else:
        fund_components.append(5)

    fundamental_score = sum(fund_components) / 100.0

    # === MARKET SCORE (20% weight) ===

    sp500_above_200dma = features.get("sp500_above_200dma", True)
    sp500_trend = features.get("sp500_trend_4wk", 0.0)
    price_vs_200dma = features.get("price_vs_200dma", 0.0)

    market_uptrend = sp500_above_200dma and sp500_trend >= -2.0

    market_components = []

    # Market trend (0-50 points)
    if sp500_above_200dma and sp500_trend >= 2:
        market_components.append(50)
    elif sp500_above_200dma and sp500_trend >= 0:
        market_components.append(40)
    elif sp500_above_200dma:
        market_components.append(25)
    else:
        market_components.append(10)

    # Stock vs 200dma (0-50 points)
    if price_vs_200dma >= 20:
        market_components.append(50)
    elif price_vs_200dma >= 10:
        market_components.append(40)
    elif price_vs_200dma >= 0:
        market_components.append(30)
    else:
        market_components.append(10)

    market_score = sum(market_components) / 100.0

    # === TOTAL SCORE ===

    total_score = (
        technical_score * 0.40 +
        fundamental_score * 0.40 +
        market_score * 0.20
    )

    return QualityScore(
        total_score=total_score,
        technical_score=technical_score,
        fundamental_score=fundamental_score,
        market_score=market_score,
        has_prior_uptrend=has_prior_uptrend,
        has_volume_confirmation=has_volume_confirmation,
        has_tight_action=has_tight_action,
        rs_at_high=rs_at_high,
        eps_growing=eps_growing,
        revenue_growing=revenue_growing,
        institutional_support=institutional_support,
        market_uptrend=market_uptrend,
        prior_uptrend_pct=prior_uptrend_pct,
        volume_ratio=volume_ratio,
        tightness=tightness,
        rs_rank=rs_rank,
        eps_growth=eps_growth,
        revenue_growth=revenue_growth,
    )


def _check_prior_uptrend(
    df: pd.DataFrame,
    base_start_date: str,
    lookback_months: int = 12,
) -> float:
    """Check for prior uptrend before pattern formation.

    Args:
        df: Stock price DataFrame.
        base_start_date: When the pattern started.
        lookback_months: How far back to check for uptrend.

    Returns:
        Percentage gain in the prior period.
    """
    try:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        base_start = pd.to_datetime(base_start_date)

        # Get data before pattern started
        prior_df = df[df["date"] < base_start].copy().reset_index(drop=True)

        if len(prior_df) < 60:  # Need at least 3 months of data
            return 0.0

        # Look at 6-12 month period before pattern
        lookback_days = lookback_months * 21  # ~21 trading days per month
        start_idx = max(0, len(prior_df) - lookback_days)

        prior_period = prior_df.iloc[start_idx:].reset_index(drop=True)
        if len(prior_period) < 60:
            return 0.0

        # Find the lowest point in the prior period
        low_price = prior_period["low"].min()
        low_idx = prior_period["low"].idxmin()

        # Get data after the low
        after_low = prior_period.iloc[low_idx:]

        if len(after_low) < 5:
            return 0.0

        high_price = after_low["high"].max()

        # Calculate gain from low to high
        if low_price > 0:
            gain_pct = ((high_price - low_price) / low_price) * 100
        else:
            gain_pct = 0.0

        return gain_pct

    except Exception as e:
        logger.warning(f"Error calculating prior uptrend: {e}")
        return 0.0


def filter_by_quality(
    patterns: list,
    stock_df: pd.DataFrame,
    index_df: pd.DataFrame,
    min_score: float = 0.6,
    min_canslim_criteria: int = 4,
) -> list:
    """Filter patterns by quality score.

    Args:
        patterns: List of DetectedPattern objects.
        stock_df: Stock price DataFrame.
        index_df: Index price DataFrame.
        min_score: Minimum quality score (0-1).
        min_canslim_criteria: Minimum CAN SLIM criteria to pass.

    Returns:
        List of patterns that pass quality filters.
    """
    from scanner.features import extract_all_features

    filtered = []

    for pattern in patterns:
        try:
            # Extract features
            features = extract_all_features(
                symbol=pattern.symbol,
                stock_df=stock_df,
                index_df=index_df,
                base_start_date=pattern.base_start_date,
                base_end_date=pattern.base_end_date,
                breakout_date=pattern.pivot_date,
                pattern_metadata=pattern.metadata,
            )

            # Compute quality score
            quality = compute_quality_score(
                stock_df=stock_df,
                index_df=index_df,
                features=features,
                base_start_date=pattern.base_start_date,
                base_end_date=pattern.base_end_date,
            )

            # Check if passes thresholds
            if quality.passes_minimum(min_score) or quality.passes_canslim(min_canslim_criteria):
                # Attach quality score to pattern metadata
                pattern.metadata["quality_score"] = quality.total_score
                pattern.metadata["technical_score"] = quality.technical_score
                pattern.metadata["fundamental_score"] = quality.fundamental_score
                pattern.metadata["prior_uptrend_pct"] = quality.prior_uptrend_pct
                filtered.append(pattern)

        except Exception as e:
            logger.warning("Error scoring pattern %s: %s", pattern.symbol, e)
            continue

    return filtered


def get_quality_summary(quality: QualityScore) -> str:
    """Generate human-readable quality summary."""
    lines = [
        f"Quality Score: {quality.total_score:.0%}",
        f"  Technical: {quality.technical_score:.0%}",
        f"  Fundamental: {quality.fundamental_score:.0%}",
        f"  Market: {quality.market_score:.0%}",
        "",
        "CAN SLIM Criteria:",
        f"  Prior Uptrend: {'PASS' if quality.has_prior_uptrend else 'FAIL'} ({quality.prior_uptrend_pct:.0f}%)",
        f"  Volume Confirm: {'PASS' if quality.has_volume_confirmation else 'FAIL'} ({quality.volume_ratio:.1f}x)",
        f"  Tight Action: {'PASS' if quality.has_tight_action else 'FAIL'} ({quality.tightness:.0%})",
        f"  RS at High: {'PASS' if quality.rs_at_high else 'FAIL'} (rank {quality.rs_rank:.0f}%)",
        f"  EPS Growth: {'PASS' if quality.eps_growing else 'FAIL'} ({quality.eps_growth:.0f}%)",
        f"  Revenue Growth: {'PASS' if quality.revenue_growing else 'FAIL'} ({quality.revenue_growth:.0f}%)",
        f"  Institutional: {'PASS' if quality.institutional_support else 'FAIL'}",
        f"  Market Uptrend: {'PASS' if quality.market_uptrend else 'FAIL'}",
    ]
    return "\n".join(lines)
