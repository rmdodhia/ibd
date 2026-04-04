"""Double Bottom (W-shape) pattern detector.

Detects:
1. Peak → decline to first low
2. Rally to mid-peak
3. Decline to second low (within 5% of first, or slight undercut)
4. Pivot = mid-peak price
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from scanner.config import get
from scanner.patterns.base_detector import BaseDetector, DetectedPattern

logger = logging.getLogger(__name__)


class DoubleBottomDetector(BaseDetector):
    """Detect double-bottom (W-shape) patterns in price data."""

    def __init__(self):
        super().__init__()
        self.min_depth = get("patterns.double_bottom.min_depth_pct", 12)
        self.max_depth = get("patterns.double_bottom.max_depth_pct", 35)
        self.min_duration_weeks = get("patterns.double_bottom.min_duration_weeks", 7)
        self.second_low_tolerance = get("patterns.double_bottom.second_low_tolerance_pct", 5)

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
        lows = df["low"].values

        # Find troughs as potential first bottoms
        trough_indices = self.find_troughs(prices, order=10)

        for first_low_idx in trough_indices:
            pattern = self._check_double_bottom(symbol, df, first_low_idx)
            if pattern:
                patterns.append(pattern)

        return patterns

    def _check_double_bottom(
        self, symbol: str, df: pd.DataFrame, first_low_idx: int
    ) -> Optional[DetectedPattern]:
        """Check if a trough forms the first low of a double bottom.

        Args:
            symbol: Stock ticker.
            df: Price DataFrame.
            first_low_idx: Index of the first low.

        Returns:
            DetectedPattern if valid, None otherwise.
        """
        prices = df["close"].values
        lows = df["low"].values
        highs = df["high"].values

        # Need prior peak for left lip
        if first_low_idx < 10:
            return None

        prior_region = prices[max(0, first_low_idx - 50) : first_low_idx]
        if len(prior_region) < 10:
            return None

        left_lip_offset = np.argmax(prior_region)
        left_lip_idx = max(0, first_low_idx - 50) + left_lip_offset
        left_lip_price = prices[left_lip_idx]
        first_low_price = lows[first_low_idx]

        # Check depth
        depth_pct = self.compute_depth_pct(left_lip_price, first_low_price)
        if depth_pct < self.min_depth or depth_pct > self.max_depth:
            return None

        # Look for mid-peak (rally after first low)
        min_mid_peak_days = 5
        max_mid_peak_days = 50

        mid_peak_search_end = min(first_low_idx + max_mid_peak_days, len(df))
        if first_low_idx + min_mid_peak_days >= len(df):
            return None

        mid_region = prices[first_low_idx:mid_peak_search_end]
        if len(mid_region) < min_mid_peak_days:
            return None

        mid_peak_offset = np.argmax(mid_region)
        mid_peak_idx = first_low_idx + mid_peak_offset
        mid_peak_price = prices[mid_peak_idx]

        # Mid-peak should be significant rally (at least 50% recovery)
        recovery_from_low = ((mid_peak_price - first_low_price) / (left_lip_price - first_low_price)) * 100
        if recovery_from_low < 40:
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

        # Second low should be within tolerance of first low
        low_diff_pct = abs((second_low_price - first_low_price) / first_low_price) * 100
        if low_diff_pct > self.second_low_tolerance:
            # Allow slight undercut (second low can be lower)
            undercut_pct = ((first_low_price - second_low_price) / first_low_price) * 100
            if undercut_pct < 0 or undercut_pct > self.second_low_tolerance:
                return None

        # Check duration
        duration_days = second_low_idx - left_lip_idx
        duration_weeks = self.trading_days_to_weeks(duration_days)
        if duration_weeks < self.min_duration_weeks:
            return None

        # Pivot is the mid-peak high
        pivot_price = highs[mid_peak_idx]

        return DetectedPattern(
            symbol=symbol,
            pattern_type="double_bottom",
            base_start_date=self._date_to_str(df.iloc[left_lip_idx]["date"]),
            base_end_date=self._date_to_str(df.iloc[second_low_idx]["date"]),
            pivot_date=self._date_to_str(df.iloc[second_low_idx]["date"]),
            pivot_price=float(pivot_price),
            confidence=self._compute_confidence(
                depth_pct, duration_weeks, low_diff_pct, recovery_from_low
            ),
            metadata={
                "depth_pct": float(depth_pct),
                "duration_weeks": float(duration_weeks),
                "first_low_price": float(first_low_price),
                "second_low_price": float(second_low_price),
                "mid_peak_price": float(mid_peak_price),
                "low_diff_pct": float(low_diff_pct),
                "recovery_pct": float(recovery_from_low),
            },
        )

    def _compute_confidence(
        self,
        depth_pct: float,
        duration_weeks: float,
        low_diff_pct: float,
        recovery_pct: float,
    ) -> float:
        """Compute confidence score for the pattern (0-1)."""
        score = 0.5

        # Ideal depth is 15-25%
        if 15 <= depth_pct <= 25:
            score += 0.15
        elif 12 <= depth_pct <= 35:
            score += 0.05

        # Second low very close to first is better
        if low_diff_pct < 2:
            score += 0.15
        elif low_diff_pct < 5:
            score += 0.1

        # Good recovery to mid-peak
        if recovery_pct >= 60:
            score += 0.1

        # Duration in typical range
        if 7 <= duration_weeks <= 20:
            score += 0.1

        return min(score, 1.0)

    @staticmethod
    def _date_to_str(date) -> str:
        """Convert date to string."""
        if hasattr(date, "strftime"):
            return date.strftime("%Y-%m-%d")
        return str(date)[:10]
