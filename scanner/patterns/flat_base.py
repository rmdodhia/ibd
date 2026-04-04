"""Flat Base pattern detector.

Detects sideways consolidation:
1. Price range from high to low ≤ 15%
2. Duration ≥ 5 weeks
3. Pivot = high of base
4. Tight weekly closes preferred
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from scanner.config import get
from scanner.patterns.base_detector import BaseDetector, DetectedPattern

logger = logging.getLogger(__name__)


class FlatBaseDetector(BaseDetector):
    """Detect flat base (tight consolidation) patterns in price data."""

    def __init__(self):
        super().__init__()
        self.max_depth = get("patterns.flat_base.max_depth_pct", 15)
        self.min_duration_weeks = get("patterns.flat_base.min_duration_weeks", 5)

    def detect(self, symbol: str, df: pd.DataFrame) -> list[DetectedPattern]:
        """Scan for flat base patterns.

        Args:
            symbol: Stock ticker symbol.
            df: DataFrame with columns [date, open, high, low, close, volume].

        Returns:
            List of detected flat base patterns.
        """
        min_days = self.min_duration_weeks * 5
        if len(df) < min_days + 10:
            return []

        patterns = []
        df = df.sort_values("date").reset_index(drop=True)

        # Slide a window to find flat regions
        window_size = min_days
        max_window = min_days * 3  # Look for bases up to 15 weeks

        for start_idx in range(0, len(df) - min_days, 5):  # Step by 1 week
            pattern = self._check_flat_base(symbol, df, start_idx, window_size, max_window)
            if pattern:
                patterns.append(pattern)

        # Deduplicate overlapping patterns
        patterns = self._dedupe_patterns(patterns)

        return patterns

    def _check_flat_base(
        self,
        symbol: str,
        df: pd.DataFrame,
        start_idx: int,
        min_window: int,
        max_window: int,
    ) -> Optional[DetectedPattern]:
        """Check if a region forms a valid flat base.

        Args:
            symbol: Stock ticker.
            df: Price DataFrame.
            start_idx: Starting index to check.
            min_window: Minimum window size (days).
            max_window: Maximum window size (days).

        Returns:
            DetectedPattern if valid, None otherwise.
        """
        # Try increasing window sizes to find the largest valid flat base
        best_pattern = None
        best_duration = 0

        for window in range(min_window, min(max_window, len(df) - start_idx) + 1, 5):
            end_idx = start_idx + window
            if end_idx > len(df):
                break

            region = df.iloc[start_idx:end_idx]
            high = region["high"].max()
            low = region["low"].min()

            if low <= 0:
                continue

            range_pct = ((high - low) / low) * 100

            if range_pct <= self.max_depth:
                duration_weeks = self.trading_days_to_weeks(window)

                if duration_weeks >= self.min_duration_weeks and duration_weeks > best_duration:
                    # Calculate tightness score
                    tightness = self._compute_tightness(region)

                    best_pattern = DetectedPattern(
                        symbol=symbol,
                        pattern_type="flat_base",
                        base_start_date=self._date_to_str(region.iloc[0]["date"]),
                        base_end_date=self._date_to_str(region.iloc[-1]["date"]),
                        pivot_date=self._date_to_str(region.iloc[-1]["date"]),
                        pivot_price=float(high),
                        confidence=self._compute_confidence(range_pct, duration_weeks, tightness),
                        metadata={
                            "depth_pct": float(range_pct),
                            "duration_weeks": float(duration_weeks),
                            "high": float(high),
                            "low": float(low),
                            "tightness_score": float(tightness),
                        },
                    )
                    best_duration = duration_weeks
            else:
                # Range too wide, stop expanding
                break

        return best_pattern

    def _compute_tightness(self, region: pd.DataFrame) -> float:
        """Compute tightness score based on weekly close variations.

        Tighter closes = higher score (0-1).
        """
        closes = region["close"].values
        if len(closes) < 5:
            return 0.5

        # Weekly closes (every 5 days)
        weekly_closes = closes[::5]
        if len(weekly_closes) < 2:
            return 0.5

        # Calculate coefficient of variation
        mean_close = np.mean(weekly_closes)
        std_close = np.std(weekly_closes)

        if mean_close == 0:
            return 0.5

        cv = std_close / mean_close

        # Convert to tightness score (lower CV = higher tightness)
        # CV < 2% is very tight, CV > 10% is loose
        tightness = max(0, min(1, 1 - (cv / 0.10)))

        return tightness

    def _compute_confidence(
        self, range_pct: float, duration_weeks: float, tightness: float
    ) -> float:
        """Compute confidence score for the pattern (0-1)."""
        score = 0.5

        # Tighter range is better
        if range_pct < 10:
            score += 0.2
        elif range_pct < 15:
            score += 0.1

        # Longer duration is better (up to a point)
        if 5 <= duration_weeks <= 8:
            score += 0.15
        elif duration_weeks > 8:
            score += 0.1

        # High tightness is better
        score += tightness * 0.15

        return min(score, 1.0)

    def _dedupe_patterns(self, patterns: list[DetectedPattern]) -> list[DetectedPattern]:
        """Remove overlapping patterns, keeping the one with higher confidence."""
        if len(patterns) <= 1:
            return patterns

        # Sort by start date
        patterns = sorted(patterns, key=lambda p: p.base_start_date)

        result = [patterns[0]]
        for p in patterns[1:]:
            last = result[-1]

            # Check for overlap (if new pattern starts before last ends)
            if p.base_start_date < last.base_end_date:
                # Keep the one with higher confidence
                if p.confidence > last.confidence:
                    result[-1] = p
            else:
                result.append(p)

        return result

    @staticmethod
    def _date_to_str(date) -> str:
        """Convert date to string."""
        if hasattr(date, "strftime"):
            return date.strftime("%Y-%m-%d")
        return str(date)[:10]
