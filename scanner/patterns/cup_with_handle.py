"""Cup with Handle pattern detector.

IBD-style cup-with-handle requirements:
1. Prior uptrend of 25%+ before the base
2. Peak (left lip) → decline 12-35%
3. Rounded bottom (not V-shaped)
4. Recovery to upper half of base (right lip)
5. Handle pullback in upper half, 10-12% max depth
6. Pivot = high of handle

Can also detect cups without handles as a separate pattern type.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from scanner.config import get
from scanner.patterns.base_detector import BaseDetector, DetectedPattern

logger = logging.getLogger(__name__)


class CupWithHandleDetector(BaseDetector):
    """Detect cup-with-handle patterns in price data."""

    def __init__(self):
        super().__init__()
        # Load config thresholds
        self.min_depth = get("patterns.cup_with_handle.min_depth_pct", 12)
        self.max_depth = get("patterns.cup_with_handle.max_depth_pct", 35)
        self.min_duration_weeks = get("patterns.cup_with_handle.min_duration_weeks", 7)
        self.max_duration_weeks = get("patterns.cup_with_handle.max_duration_weeks", 65)
        self.handle_max_depth = get("patterns.cup_with_handle.handle_max_depth_pct", 12)
        self.handle_min_weeks = get("patterns.cup_with_handle.handle_min_weeks", 1)
        self.handle_max_weeks = get("patterns.cup_with_handle.handle_max_weeks", 4)

        # Right lip upper-half requirement
        self.right_lip_upper_half = get(
            "patterns.cup_with_handle.right_lip_upper_half", True
        )
        self.right_lip_min_recovery = get(
            "patterns.cup_with_handle.right_lip_min_recovery_pct", 50
        )

        # Roundness (now configurable)
        self.roundness_min_pct = get(
            "patterns.cup_with_handle.roundness_min_pct", 10
        )
        self.roundness_min_symmetry = get(
            "patterns.cup_with_handle.roundness_min_symmetry", 0.25
        )

        # Handle requirement
        self.require_handle = get("patterns.cup_with_handle.require_handle", False)

        # Prior uptrend
        self.require_prior_uptrend = get("patterns.prior_uptrend.enabled", True)
        self.min_prior_advance = get("patterns.prior_uptrend.min_advance_pct", 25)
        self.prior_lookback_weeks = get("patterns.prior_uptrend.lookback_weeks", 26)

    def detect(self, symbol: str, df: pd.DataFrame) -> list[DetectedPattern]:
        """Scan for cup-with-handle patterns.

        Args:
            symbol: Stock ticker symbol.
            df: DataFrame with columns [date, open, high, low, close, volume].

        Returns:
            List of detected cup-with-handle patterns.
        """
        if len(df) < self.min_duration_weeks * 5 + 20:
            return []

        patterns = []
        df = df.sort_values("date").reset_index(drop=True)
        prices = df["close"].values

        # Find peaks as potential left lips (order=7 for ~1.5 week window)
        peak_indices = self.find_peaks(prices, order=7)

        for left_lip_idx in peak_indices:
            pattern = self._check_cup_from_peak(symbol, df, left_lip_idx)
            if pattern:
                patterns.append(pattern)

        return patterns

    def _check_cup_from_peak(
        self, symbol: str, df: pd.DataFrame, left_lip_idx: int
    ) -> Optional[DetectedPattern]:
        """Check if a peak forms the left lip of a valid cup.

        Args:
            symbol: Stock ticker.
            df: Price DataFrame.
            left_lip_idx: Index of the potential left lip peak.

        Returns:
            DetectedPattern if valid, None otherwise.
        """
        prices = df["close"].values
        left_lip_price = prices[left_lip_idx]

        min_duration_days = self.min_duration_weeks * 5
        max_duration_days = self.max_duration_weeks * 5

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

        # Look for the cup bottom
        search_end = min(left_lip_idx + max_duration_days, len(df))
        if left_lip_idx + min_duration_days >= len(df):
            return None

        # Find the lowest point after left lip
        cup_region = prices[left_lip_idx : search_end]
        if len(cup_region) < min_duration_days:
            return None

        trough_offset = np.argmin(cup_region)
        trough_idx = left_lip_idx + trough_offset
        trough_price = prices[trough_idx]

        # Check depth
        depth_pct = self.compute_depth_pct(left_lip_price, trough_price)
        if depth_pct < self.min_depth or depth_pct > self.max_depth:
            return None

        # Look for right lip (recovery to upper half of base)
        right_search_start = trough_idx + 5
        right_search_end = min(trough_idx + max_duration_days // 2, len(df))

        if right_search_start >= len(df):
            return None

        right_region = prices[right_search_start:right_search_end]
        if len(right_region) < 5:
            return None

        # Right lip is the highest point after trough
        right_lip_offset = np.argmax(right_region)
        right_lip_idx = right_search_start + right_lip_offset
        right_lip_price = prices[right_lip_idx]

        # IBD REQUIREMENT: Right lip must be in upper half of base
        # Formula: right_lip >= trough + 0.5 * (left_lip - trough)
        # This is equivalent to: recovery >= 50% of the decline
        if self.right_lip_upper_half:
            min_right_lip = trough_price + (
                self.right_lip_min_recovery / 100
            ) * (left_lip_price - trough_price)
            if right_lip_price < min_right_lip:
                return None

        recovery_pct = (
            (right_lip_price - trough_price) / (left_lip_price - trough_price)
        ) * 100

        # Check for handle (optional based on config)
        handle_result = self._find_handle(df, right_lip_idx, right_lip_price)

        if handle_result is not None:
            handle_end_idx, handle_depth_pct, pivot_price = handle_result
            has_handle = True
        else:
            # Cup without handle
            if self.require_handle:
                return None  # Skip if handle is required

            handle_end_idx = right_lip_idx
            handle_depth_pct = 0.0
            pivot_price = right_lip_price
            has_handle = False

        # Check cup duration
        cup_duration_days = handle_end_idx - left_lip_idx
        cup_duration_weeks = self.trading_days_to_weeks(cup_duration_days)
        if (
            cup_duration_weeks < self.min_duration_weeks
            or cup_duration_weeks > self.max_duration_weeks
        ):
            return None

        # Check roundness (not V-shaped)
        if not self._check_roundness(prices, left_lip_idx, trough_idx, right_lip_idx):
            return None

        # Determine pattern type
        pattern_type = "cup_with_handle" if has_handle else "cup_without_handle"

        # Build pattern
        base_start = df.iloc[left_lip_idx]["date"]
        base_end = df.iloc[handle_end_idx]["date"]
        pivot_date = df.iloc[handle_end_idx]["date"]

        return DetectedPattern(
            symbol=symbol,
            pattern_type=pattern_type,
            base_start_date=self._date_to_str(base_start),
            base_end_date=self._date_to_str(base_end),
            pivot_date=self._date_to_str(pivot_date),
            pivot_price=float(pivot_price),
            confidence=self._compute_confidence(
                depth_pct,
                cup_duration_weeks,
                handle_depth_pct,
                recovery_pct,
                has_handle,
                advance_pct,
            ),
            metadata={
                "depth_pct": float(depth_pct),
                "duration_weeks": float(cup_duration_weeks),
                "handle_depth_pct": float(handle_depth_pct),
                "recovery_pct": float(recovery_pct),
                "left_lip_price": float(left_lip_price),
                "trough_price": float(trough_price),
                "right_lip_price": float(right_lip_price),
                "has_handle": has_handle,
                "prior_advance_pct": float(advance_pct),
            },
        )

    def _find_handle(
        self, df: pd.DataFrame, right_lip_idx: int, right_lip_price: float
    ) -> Optional[tuple[int, float, float]]:
        """Find a valid handle after the right lip.

        Handle must form in upper half of base and have limited depth.

        Returns:
            Tuple of (handle_end_idx, handle_depth_pct, pivot_price) or None.
        """
        prices = df["close"].values
        lows = df["low"].values
        highs = df["high"].values

        handle_min_days = self.handle_min_weeks * 5
        handle_max_days = self.handle_max_weeks * 5

        handle_end = min(right_lip_idx + handle_max_days, len(df))
        if right_lip_idx + handle_min_days >= len(df):
            return None

        handle_region_low = lows[right_lip_idx:handle_end]
        handle_region_high = highs[right_lip_idx:handle_end]

        if len(handle_region_low) < handle_min_days:
            return None

        # Handle low should be within allowed depth
        handle_low = np.min(handle_region_low)
        handle_depth = self.compute_depth_pct(right_lip_price, handle_low)

        # Reject if handle is too deep
        if handle_depth > self.handle_max_depth:
            return None

        # Handle must have some pullback (at least 2%)
        if handle_depth < 2:
            return None

        # Pivot is the high of the handle
        pivot_price = np.max(handle_region_high)
        handle_end_idx = min(right_lip_idx + len(handle_region_low) - 1, len(df) - 1)

        return handle_end_idx, handle_depth, pivot_price

    def _check_roundness(
        self, prices: np.ndarray, left_idx: int, trough_idx: int, right_idx: int
    ) -> bool:
        """Check if the cup has a rounded bottom (not V-shaped).

        A rounded cup should spend time near the bottom, not immediately reverse.
        Uses configurable thresholds.
        """
        # Time from left lip to trough vs trough to right lip
        left_duration = trough_idx - left_idx
        right_duration = right_idx - trough_idx

        # Should be reasonably symmetric
        if left_duration <= 0 or right_duration <= 0:
            return False

        symmetry = min(left_duration, right_duration) / max(
            left_duration, right_duration
        )

        # Check symmetry against configurable threshold
        if symmetry < self.roundness_min_symmetry:
            return False

        # Check for V-shape by looking at how many days spent near trough
        cup_prices = prices[left_idx : right_idx + 1]
        trough_price = prices[trough_idx]
        cup_depth = prices[left_idx] - trough_price

        if cup_depth <= 0:
            return False

        # Days within 20% of trough depth
        near_trough = np.sum(cup_prices <= trough_price + cup_depth * 0.2)
        roundness_pct = (near_trough / len(cup_prices)) * 100

        # Check roundness against configurable threshold
        return roundness_pct >= self.roundness_min_pct

    def _compute_confidence(
        self,
        depth_pct: float,
        duration_weeks: float,
        handle_depth_pct: float,
        recovery_pct: float,
        has_handle: bool,
        advance_pct: float,
    ) -> float:
        """Compute confidence score for the pattern (0-1)."""
        score = 0.5

        # Ideal depth is 15-25%
        if 15 <= depth_pct <= 25:
            score += 0.12
        elif 12 <= depth_pct <= 35:
            score += 0.06

        # Ideal duration is 7-30 weeks
        if 7 <= duration_weeks <= 30:
            score += 0.08

        # Cup with handle is stronger than cup without
        if has_handle:
            score += 0.1
            # Handle depth < 10% is better
            if handle_depth_pct < 10:
                score += 0.08
            elif handle_depth_pct < 12:
                score += 0.04

        # Good recovery (right lip in upper part of base)
        if recovery_pct >= 70:
            score += 0.1
        elif recovery_pct >= 50:
            score += 0.05

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
