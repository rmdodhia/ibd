"""Double Bottom (W-shape) pattern detector.

IBD-style double bottom requirements:
1. Prior uptrend of 25%+ before the base
2. Peak → decline to first low (12-35% depth)
3. Rally to mid-peak (must be meaningful - 10%+ above both lows)
4. Decline to second low that UNDERCUTS first low (0.5-5% below)
5. Pivot = mid-peak price

The undercut is a key IBD requirement - it shakes out weak holders.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from scanner.config import get, get_price_high_low_arrays
from scanner.patterns.base_detector import BaseDetector, DetectedPattern

logger = logging.getLogger(__name__)


class DoubleBottomDetector(BaseDetector):
    """Detect double-bottom (W-shape) patterns in price data."""

    def __init__(self):
        super().__init__()
        # Depth and duration
        self.min_depth = get("patterns.double_bottom.min_depth_pct", 12)
        self.max_depth = get("patterns.double_bottom.max_depth_pct", 35)
        self.min_duration_weeks = get("patterns.double_bottom.min_duration_weeks", 5)

        # Second low undercut requirements (IBD: must undercut)
        self.second_low_min_undercut = get(
            "patterns.double_bottom.second_low_min_undercut_pct", 0.5
        )
        self.second_low_max_undercut = get(
            "patterns.double_bottom.second_low_max_undercut_pct", 5.0
        )

        # Mid-peak prominence requirement
        self.mid_peak_min_rise = get(
            "patterns.double_bottom.mid_peak_min_rise_pct", 10.0
        )

        # Minimum time between lows
        self.min_weeks_between_lows = get(
            "patterns.double_bottom.min_weeks_between_lows", 2
        )

        # Prior uptrend
        self.require_prior_uptrend = get("patterns.prior_uptrend.enabled", True)
        self.min_prior_advance = get("patterns.prior_uptrend.min_advance_pct", 25)
        self.prior_lookback_weeks = get("patterns.prior_uptrend.lookback_weeks", 26)

    def detect(self, symbol: str, df: pd.DataFrame) -> list[DetectedPattern]:
        """Scan for double-bottom patterns.

        Args:
            symbol: Stock ticker symbol.
            df: DataFrame with columns [date, open, high, low, close, volume].

        Returns:
            List of detected double-bottom patterns.
        """
        if len(df) < self.min_duration_weeks * 5 + 20:
            return []

        patterns = []
        df = df.sort_values("date").reset_index(drop=True)
        prices = df["close"].values

        # Find troughs as potential first bottoms (order=5 for balanced detection)
        trough_indices = self.find_troughs(prices, order=5)

        for first_low_idx in trough_indices:
            pattern = self._check_double_bottom(symbol, df, first_low_idx)
            if pattern:
                patterns.append(pattern)

        return patterns

    def _check_double_bottom(
        self, symbol: str, df: pd.DataFrame, first_low_idx: int
    ) -> Optional[DetectedPattern]:
        """Check if a trough forms the first low of a valid IBD double bottom.

        Args:
            symbol: Stock ticker.
            df: Price DataFrame.
            first_low_idx: Index of the first low.

        Returns:
            DetectedPattern if valid, None otherwise.
        """
        prices = df["close"].values
        highs, lows = get_price_high_low_arrays(df)

        # Need prior peak for left lip
        if first_low_idx < 10:
            return None

        prior_region = prices[max(0, first_low_idx - 70) : first_low_idx]
        if len(prior_region) < 10:
            return None

        left_lip_offset = np.argmax(prior_region)
        left_lip_idx = max(0, first_low_idx - 70) + left_lip_offset
        left_lip_price = prices[left_lip_idx]
        first_low_price = lows[first_low_idx]

        # Check prior uptrend requirement
        if self.require_prior_uptrend:
            has_uptrend, advance_pct = self.check_prior_uptrend(
                df,
                left_lip_idx,
                min_advance_pct=self.min_prior_advance,
                lookback_weeks=self.prior_lookback_weeks,
            )
            if not has_uptrend:
                return None
        else:
            advance_pct = 0.0

        # Check depth
        depth_pct = self.compute_depth_pct(left_lip_price, first_low_price)
        if depth_pct < self.min_depth or depth_pct > self.max_depth:
            return None

        # Look for mid-peak (rally after first low)
        min_mid_peak_days = self.min_weeks_between_lows * 5  # At least 2 weeks
        max_mid_peak_days = 70

        mid_peak_search_end = min(first_low_idx + max_mid_peak_days, len(df))
        if first_low_idx + min_mid_peak_days >= len(df):
            return None

        mid_region = prices[first_low_idx:mid_peak_search_end]
        if len(mid_region) < min_mid_peak_days:
            return None

        mid_peak_offset = np.argmax(mid_region)
        mid_peak_idx = first_low_idx + mid_peak_offset
        mid_peak_price = prices[mid_peak_idx]

        # Mid-peak must be meaningful: at least X% above the first low
        mid_peak_rise_pct = ((mid_peak_price - first_low_price) / first_low_price) * 100
        if mid_peak_rise_pct < self.mid_peak_min_rise:
            return None

        # Look for second low
        second_low_search_end = min(mid_peak_idx + max_mid_peak_days, len(df))
        if mid_peak_idx + min_mid_peak_days >= len(df):
            return None

        second_region_low = lows[mid_peak_idx:second_low_search_end]
        if len(second_region_low) < min_mid_peak_days:
            return None

        second_low_offset = np.argmin(second_region_low)
        second_low_idx = mid_peak_idx + second_low_offset
        second_low_price = lows[second_low_idx]

        # IBD REQUIREMENT: Second low must UNDERCUT first low (shake out weak holders)
        # Calculate how much the second low is below the first low
        undercut_pct = ((first_low_price - second_low_price) / first_low_price) * 100

        # Must undercut by at least min amount
        if undercut_pct < self.second_low_min_undercut:
            return None

        # But not too deep (would be a breakdown, not a W)
        if undercut_pct > self.second_low_max_undercut:
            return None

        # Verify mid-peak is also above the second low
        mid_peak_above_second = (
            (mid_peak_price - second_low_price) / second_low_price
        ) * 100
        if mid_peak_above_second < self.mid_peak_min_rise:
            return None

        # Check minimum time between lows
        days_between_lows = second_low_idx - first_low_idx
        weeks_between_lows = self.trading_days_to_weeks(days_between_lows)
        if weeks_between_lows < self.min_weeks_between_lows:
            return None

        # Check total duration
        duration_days = second_low_idx - left_lip_idx
        duration_weeks = self.trading_days_to_weeks(duration_days)
        if duration_weeks < self.min_duration_weeks:
            return None

        # Pivot is the mid-peak high
        pivot_price = highs[mid_peak_idx]

        # Find when price breaks above pivot after second low (actual breakout point)
        breakout_idx = second_low_idx
        for i in range(second_low_idx + 1, min(second_low_idx + 30, len(df))):
            if prices[i] >= pivot_price:
                breakout_idx = i
                break

        return DetectedPattern(
            symbol=symbol,
            pattern_type="double_bottom",
            base_start_date=self._date_to_str(df.iloc[left_lip_idx]["date"]),
            base_end_date=self._date_to_str(df.iloc[breakout_idx]["date"]),
            pivot_date=self._date_to_str(df.iloc[breakout_idx]["date"]),
            pivot_price=float(pivot_price),
            confidence=self._compute_confidence(
                depth_pct,
                duration_weeks,
                undercut_pct,
                mid_peak_rise_pct,
                advance_pct,
            ),
            metadata={
                "depth_pct": float(depth_pct),
                "duration_weeks": float(duration_weeks),
                "first_low_price": float(first_low_price),
                "second_low_price": float(second_low_price),
                "mid_peak_price": float(mid_peak_price),
                "undercut_pct": float(undercut_pct),
                "mid_peak_rise_pct": float(mid_peak_rise_pct),
                "weeks_between_lows": float(weeks_between_lows),
                "prior_advance_pct": float(advance_pct),
            },
        )

    def _compute_confidence(
        self,
        depth_pct: float,
        duration_weeks: float,
        undercut_pct: float,
        mid_peak_rise_pct: float,
        advance_pct: float,
    ) -> float:
        """Compute confidence score for the pattern (0-1)."""
        score = 0.5

        # Ideal depth is 15-25%
        if 15 <= depth_pct <= 25:
            score += 0.15
        elif 12 <= depth_pct <= 35:
            score += 0.08

        # Ideal undercut is 1-3% (classic shakeout)
        if 1 <= undercut_pct <= 3:
            score += 0.15
        elif 0.5 <= undercut_pct <= 5:
            score += 0.08

        # Strong mid-peak rally is better
        if mid_peak_rise_pct >= 15:
            score += 0.1
        elif mid_peak_rise_pct >= 10:
            score += 0.05

        # Duration in typical range
        if 5 <= duration_weeks <= 15:
            score += 0.1

        # Strong prior uptrend
        if advance_pct >= 40:
            score += 0.1
        elif advance_pct >= 25:
            score += 0.05

        return min(score, 1.0)

    @staticmethod
    def _date_to_str(date) -> str:
        """Convert date to string."""
        if hasattr(date, "strftime"):
            return date.strftime("%Y-%m-%d")
        return str(date)[:10]
