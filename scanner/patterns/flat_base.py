"""Flat Base pattern detector.

IBD-style flat base requirements:
1. Prior uptrend of 25%+ before the base
2. Tight sideways consolidation: high-to-low range <= 15%
3. Duration >= 5 weeks
4. Tight weekly closes (low coefficient of variation)
5. Pivot = high of base

Flat bases are continuation patterns that form after a meaningful advance.
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

        # Tightness as hard filter (not just confidence)
        self.max_tightness_cv = get("patterns.flat_base.max_tightness_cv", 0.08)

        # Prior uptrend
        self.require_prior_uptrend = get(
            "patterns.flat_base.require_prior_uptrend", True
        )
        self.min_prior_advance = get("patterns.prior_uptrend.min_advance_pct", 25)
        self.prior_lookback_weeks = get("patterns.prior_uptrend.lookback_weeks", 26)

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
        # Check prior uptrend requirement first
        if self.require_prior_uptrend:
            has_uptrend, advance_pct = self.check_prior_uptrend(
                df,
                start_idx,
                min_advance_pct=self.min_prior_advance,
                lookback_weeks=self.prior_lookback_weeks,
            )
            if not has_uptrend:
                return None
        else:
            advance_pct = 0.0

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

                # Calculate tightness score
                tightness_cv = self._compute_tightness_cv(region)

                # HARD FILTER: Reject if too loose
                if tightness_cv > self.max_tightness_cv:
                    continue

                if (
                    duration_weeks >= self.min_duration_weeks
                    and duration_weeks > best_duration
                ):
                    tightness_score = self._cv_to_score(tightness_cv)

                    best_pattern = DetectedPattern(
                        symbol=symbol,
                        pattern_type="flat_base",
                        base_start_date=self._date_to_str(region.iloc[0]["date"]),
                        base_end_date=self._date_to_str(region.iloc[-1]["date"]),
                        pivot_date=self._date_to_str(region.iloc[-1]["date"]),
                        pivot_price=float(high),
                        confidence=self._compute_confidence(
                            range_pct, duration_weeks, tightness_score, advance_pct
                        ),
                        metadata={
                            "depth_pct": float(range_pct),
                            "duration_weeks": float(duration_weeks),
                            "high": float(high),
                            "low": float(low),
                            "tightness_cv": float(tightness_cv),
                            "tightness_score": float(tightness_score),
                            "prior_advance_pct": float(advance_pct),
                        },
                    )
                    best_duration = duration_weeks
            else:
                # Range too wide, stop expanding
                break

        return best_pattern

    def _compute_tightness_cv(self, region: pd.DataFrame) -> float:
        """Compute coefficient of variation of weekly closes.

        Lower CV = tighter consolidation.
        Returns CV as a decimal (e.g., 0.05 for 5%).
        """
        closes = region["close"].values
        if len(closes) < 5:
            return 1.0  # Return high CV for insufficient data

        # Weekly closes (every 5 days)
        weekly_closes = closes[::5]
        if len(weekly_closes) < 2:
            return 1.0

        # Calculate coefficient of variation
        mean_close = np.mean(weekly_closes)
        std_close = np.std(weekly_closes)

        if mean_close == 0:
            return 1.0

        return std_close / mean_close

    def _cv_to_score(self, cv: float) -> float:
        """Convert CV to tightness score (0-1).

        Lower CV = higher score.
        """
        # CV < 2% is very tight (score ~0.8)
        # CV > 8% is loose (score ~0)
        return max(0, min(1, 1 - (cv / 0.10)))

    def _compute_confidence(
        self,
        range_pct: float,
        duration_weeks: float,
        tightness: float,
        advance_pct: float,
    ) -> float:
        """Compute confidence score for the pattern (0-1)."""
        score = 0.5

        # Tighter range is better
        if range_pct < 10:
            score += 0.15
        elif range_pct < 15:
            score += 0.08

        # Longer duration is better (up to a point)
        if 5 <= duration_weeks <= 8:
            score += 0.1
        elif duration_weeks > 8:
            score += 0.05

        # High tightness is better
        score += tightness * 0.15

        # Strong prior uptrend
        if advance_pct >= 40:
            score += 0.1
        elif advance_pct >= 25:
            score += 0.05

        return min(score, 1.0)

    def _dedupe_patterns(
        self, patterns: list[DetectedPattern]
    ) -> list[DetectedPattern]:
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
